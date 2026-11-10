import importlib.util
import io
import os
import sys
import threading

import pytest
from rich.console import Console
from rich.syntax import Syntax

from kalfa.commands import legos
from kalfa.core import aliases
from kalfa.core.errors import KalfaError
from kalfa.records.reader import records_under, summary_of
from kalfa.tui import app as app_module
from kalfa.tui.app import KalfaApp
from kalfa.tui.jobs import cli_argv, follow, pairs_of, shown_line, spawn, words_of
from kalfa.tui.records import best_text, detail_of, number_text, row_cells, spark_series, summary_text, turn_text
from records.fixtures import make_record, make_sweep


@pytest.fixture
def root(tmp_path):
    make_record(tmp_path / "runs" / "alpha", status="done")
    make_record(tmp_path / "runs" / "beta", status="running")
    make_sweep(tmp_path / "runs" / "sweep", count=3)
    return tmp_path / "runs"


def plain(renderable):
    console = Console(width=100, record=True, file=io.StringIO())
    console.print(renderable)
    return console.export_text()


def rows_of(root):
    return {entry["path"]: row_cells(root, entry) for entry in records_under(root)}


def test_pairs_keep_quoted_values_and_split_on_spaces():
    assert pairs_of("lr=0.1 'name=a b' train.epochs=2") == ["lr=0.1", "name=a b", "train.epochs=2"]
    assert pairs_of("") == []
    assert pairs_of(None) == []


def test_pairs_reject_words_without_a_name_and_a_value():
    with pytest.raises(KalfaError, match="-p takes NAME=VALUE"):
        pairs_of("lr=0.1 width", "-p")
    with pytest.raises(KalfaError, match="got =3"):
        pairs_of("=3")
    with pytest.raises(KalfaError, match="cannot split"):
        pairs_of("name='open")


def test_words_split_config_paths_like_a_shell():
    assert words_of("a.yaml 'my dir/b.yaml'") == ["a.yaml", "my dir/b.yaml"]
    assert words_of("   ") == []


def test_cli_argv_runs_the_module_with_the_overrides():
    argv = cli_argv("run", "c.yaml", "d.yaml", sets=["train.epochs=2"], params=["lr=0.1", "width=8"])
    assert argv == [sys.executable, "-m", "kalfa.cli", "run", "c.yaml", "d.yaml", "--set", "train.epochs=2",
                    "--param", "lr=0.1", "--param", "width=8"]
    assert cli_argv("board", "runs", "--open")[3:] == ["board", "runs", "--open"]


def test_numbers_turns_and_best_read_short():
    assert number_text(0.123456) == "0.1235"
    assert number_text(None) == "-"
    assert number_text(3) == "3"
    assert turn_text({"turn": 3, "total": 10}) == "3/10"
    assert turn_text({"turn": 2, "total": None}) == "2"
    assert turn_text({}) == "-"
    assert best_text({"series": "val/rmse", "value": 0.5, "turn": 3}) == "val/rmse 0.5 @3"
    assert best_text({"series": "val/rmse", "value": None, "turn": None}) == "val/rmse -"
    assert best_text(None) == ""


def test_summary_text_names_status_best_params_and_last_values(root):
    text = summary_text(summary_of(root / "alpha"))
    lines = text.splitlines()
    assert lines[0] == "alpha"
    assert "done, turn 3/10 epochs" in text
    assert "cpu on box" in text
    assert any(line.split()[:2] == ["best", "val/rmse"] and line.endswith("(min)") for line in lines if line.strip())
    assert "params" in lines and "last" in lines
    assert lines.index("params") < lines.index("last")
    assert any(line.split() == ["lr", "0.001"] for line in lines)
    assert any(line.split()[:1] == ["train/mse"] for line in lines)
    assert "rules     to_huber" in text


def test_summary_text_shows_a_failure_and_skips_empty_blocks():
    text = summary_text({"name": "x", "status": "failed", "turn": 1, "total": 4, "failure": "boom", "params": {},
                         "last": {}})
    assert text.splitlines()[0] == "x"
    assert "failure   boom" in text
    assert "params" not in text.splitlines() and "last" not in text.splitlines()


def test_rows_cover_records_and_sweeps(root):
    rows = rows_of(root)
    assert list(rows) == ["alpha", "beta", "sweep"]
    name, status, turn, best = rows["alpha"]
    assert (name, status, turn) == ("alpha", "done", "3/10")
    assert best.startswith("val/rmse ") and best.endswith("@3")
    assert rows["beta"][1] == "stale"
    name, status, turn, best = rows["sweep"]
    assert (name, status, turn) == ("sweep", "sweep", "2/3")
    assert best.startswith("#") and " val/rmse " in best


def test_a_broken_manifest_reads_as_unreadable(tmp_path):
    make_record(tmp_path / "gamma")
    (tmp_path / "gamma" / "manifest.json").write_text("{", encoding="utf-8")
    entry = {"path": "gamma", "kind": "record"}
    assert row_cells(tmp_path, entry) == ("gamma", "unreadable", "", "")
    text, sparks = detail_of(tmp_path, entry)
    assert "not readable" in text and sparks == []


def test_the_root_itself_can_be_a_record(tmp_path):
    directory = make_record(tmp_path / "solo")
    entries = records_under(directory)
    assert entries[0]["path"] == "."
    assert row_cells(directory, entries[0])[:3] == ("solo", "done", "3/10")


def test_detail_of_a_record_has_at_most_six_sparks(root):
    text, sparks = detail_of(root, {"path": "alpha", "kind": "record"})
    assert text.startswith("alpha\n")
    names = [name for name, _ in sparks]
    assert names[:4] == ["train/mse", "val/mse", "val/rmse", "test/rmse"]
    assert "turn" not in names and "update" not in names and "minimizes/main" not in names
    assert len(sparks) <= 6
    assert all(len(values) == 3 and all(isinstance(value, float) for value in values) for _, values in sparks)
    assert len(spark_series(root / "alpha", limit=2)) == 2


def test_detail_of_a_sweep_names_its_points_and_best(root):
    text, sparks = detail_of(root, {"path": "sweep", "kind": "sweep"})
    lines = text.splitlines()
    assert lines[0] == "sweep (sweep)"
    assert "3 of 3: 2 done, 0 running, 1 failed" in text
    assert "val/rmse (min, at best)" in text
    assert any(line.split()[:2] == ["best", "point"] for line in lines)
    assert "space" in lines
    assert sparks == []


def test_shown_line_keeps_what_a_terminal_shows():
    assert shown_line(b"plain") == "plain"
    assert shown_line(b"10%\r50%\r100%\r") == "100%"
    assert shown_line("ağ".encode()) == "ağ"
    assert shown_line(b"\xff") == "�"


def test_spawn_and_follow_stream_a_process(tmp_path):
    code = "import sys\nprint('one')\nprint('two')\nsys.stdout.write('tail')\nsys.exit(3)\n"
    process, path = spawn([sys.executable, "-c", code], "test", directory=tmp_path)
    assert os.path.dirname(path) == str(tmp_path)
    lines = []
    assert follow(path, process, lines.append, threading.Event(), every=0.05, label="test") == 3
    assert lines == ["one", "two", "tail", "test exited with 3"]


def test_follow_stops_when_told(tmp_path):
    process, path = spawn([sys.executable, "-c", "import time\ntime.sleep(30)\n"], "test", directory=tmp_path)
    stop = threading.Event()
    stop.set()
    try:
        assert follow(path, process, [].append, stop) is None
    finally:
        process.kill()
        process.wait()


def test_follow_stops_when_the_page_refuses_lines(tmp_path):
    process, path = spawn([sys.executable, "-c", "print('one')\nprint('two')\n"], "test", directory=tmp_path)
    process.wait()
    seen = []

    def emit(line):
        seen.append(line)
        return False

    assert follow(path, process, emit, threading.Event()) is None
    assert seen == ["one"]


def test_the_app_asks_for_textual_when_it_is_missing(monkeypatch):
    names = {"textual", *(name for name in sys.modules if name.startswith("textual."))}
    for name in names:
        monkeypatch.setitem(sys.modules, name, None)
    spec = importlib.util.spec_from_file_location("kalfa_tui_without_textual", app_module.__file__)
    bare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bare)
    assert bare.App is None and isinstance(bare.TEXTUAL_ERROR, ImportError)
    with pytest.raises(KalfaError, match=r"needs the textual package \(uv sync installs the dependencies of kalfa\)"):
        bare.KalfaApp(root=".")
    assert app_module.App is not None and app_module.TEXTUAL_ERROR is None


@pytest.mark.asyncio
async def test_the_o_key_draws_the_plots_of_the_selected_record_through_the_cli(root, monkeypatch):
    launched = []

    def launch(self, argv, label, target, detach=True):
        launched.append((argv, label, target))

    monkeypatch.setattr(KalfaApp, "launch", launch)
    app = KalfaApp(root=str(root))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("o")
        await pilot.pause()
    assert launched == [([sys.executable, "-m", "kalfa.cli", "plots", str(root / "alpha")], "plots", "#records-log")]


@pytest.mark.asyncio
async def test_the_detail_follows_the_cursor_and_draws_sparks(root):
    app = KalfaApp(root=str(root))
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.focused is app.query_one("#records-table")
        assert str(app.query_one("#record-detail").content).startswith("alpha\n")
        await pilot.press("down")
        await pilot.pause()
        assert str(app.query_one("#record-detail").content).startswith("beta\n")
        assert len(app.query("#record-sparks Sparkline")) == 5
        await pilot.press("r")
        await pilot.pause()
        assert app.query_one("#records-table").cursor_row == 1


@pytest.mark.asyncio
async def test_record_keys_explain_a_finished_record_and_a_sweep(root):
    app = KalfaApp(root=str(root))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("s")
        await pilot.pause()
        log = app.query_one("#records-log")
        assert "stop: nothing runs under" in "\n".join(log.plain)
        app.query_one("#records-table").move_cursor(row=2)
        await pilot.pause()
        assert str(app.query_one("#record-detail").content).startswith("sweep (sweep)")
        await pilot.press("p")
        await pilot.pause()
        assert "predict works on a record and sweep is a sweep root" in "\n".join(log.plain)


@pytest.mark.asyncio
async def test_config_overrides_are_read_before_anything_runs(root, tmp_path):
    config = tmp_path / "c.yaml"
    config.write_text("train: {epochs: 1}\n", encoding="utf-8")
    app = KalfaApp(root=str(root), config=str(config))
    async with app.run_test() as pilot:
        await pilot.press("3")
        await pilot.pause()
        assert app.focused is app.query_one("#config-output")
        assert app.query_one("#config-path").value == str(config)
        app.query_one("#config-sets").value = "train.epochs"
        await pilot.press("x")
        await pilot.pause()
        assert "--set takes NAME=VALUE" in "\n".join(app.query_one("#config-output").plain)


@pytest.fixture
def std_legos(monkeypatch):
    for name, entry in list(aliases.ALIASES.items()):
        if entry.plugin:
            monkeypatch.delitem(aliases.ALIASES, name)


def branches_of(tree):
    return [str(branch.label) for branch in tree.root.children]


def leaves_of(tree):
    return [leaf.data.name for branch in tree.root.children for leaf in branch.children]


@pytest.mark.asyncio
async def test_the_legos_tree_starts_with_the_categories_closed_and_counted(root, std_legos):
    app = KalfaApp(root=str(root))
    async with app.run_test() as pilot:
        await pilot.press("2")
        await pilot.pause()
        tree = app.query_one("#legos-tree")
        assert app.focused is tree
        found = legos.categories()
        assert branches_of(tree) == [f"{category} ({len(entries)})" for category, entries in found]
        assert "losses (6)" in branches_of(tree)
        assert not any(branch.is_expanded for branch in tree.root.children)
        assert leaves_of(tree) == [entry.name for _, entries in found for entry in entries]
        mlp = next(leaf for branch in tree.root.children for leaf in branch.children if leaf.data.name == "mlp")
        assert str(mlp.label).startswith("mlp  ") and legos.flat(mlp.data.doc) in str(mlp.label)
        page = plain(app.query_one("#legos-page").content)
        assert page == plain(legos.category_renderable("data", dict(found)["data"]))
        assert page.splitlines()[0] == "data" and "table" in page


@pytest.mark.asyncio
async def test_the_legos_search_filters_the_tree_and_opens_the_categories_with_hits(root, std_legos):
    app = KalfaApp(root=str(root))
    async with app.run_test() as pilot:
        await pilot.press("2")
        await pilot.pause()
        search, tree = app.query_one("#legos-search"), app.query_one("#legos-tree")
        search.value = "scaler"
        await pilot.pause()
        assert branches_of(tree) == ["table preprocessing (3)"]
        assert all(branch.is_expanded for branch in tree.root.children)
        assert leaves_of(tree) == ["minmax_scaler", "robust_scaler", "standard_scaler"]
        assert tree.cursor_node.data.name == "minmax_scaler"
        page = plain(app.query_one("#legos-page").content)
        assert page == plain(legos.entry_renderable("minmax_scaler"))
        assert "examples:" in page and "params:" in page
        search.value = "mlp"
        await pilot.pause()
        assert "mlp" in leaves_of(tree) and all(branch.is_expanded for branch in tree.root.children)
        assert plain(app.query_one("#legos-page").content).splitlines()[0] == tree.cursor_node.data.name
        search.value = "no-such-word-anywhere"
        await pilot.pause()
        assert not tree.root.children
        assert "no name matches" in str(app.query_one("#legos-page").content)
        search.value = "adam"
        search.value = "adam no-such-word-anywhere"
        await pilot.pause()
        assert not tree.root.children
        assert "no name matches" in str(app.query_one("#legos-page").content)
        search.value = "  "
        await pilot.pause()
        assert len(tree.root.children) == len(legos.categories())
        assert not any(branch.is_expanded for branch in tree.root.children)
        search.focus()
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert app.focused is tree
        search.focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert app.focused is tree


@pytest.mark.asyncio
async def test_the_legos_page_shows_the_highlighted_leaf_with_colored_yaml(root, std_legos):
    app = KalfaApp(root=str(root))
    async with app.run_test() as pilot:
        await pilot.press("2")
        await pilot.pause()
        tree = app.query_one("#legos-tree")
        branch = next(node for node in tree.root.children if node.data == "optimizers")
        tree.move_cursor(branch)
        await pilot.pause()
        overview = plain(app.query_one("#legos-page").content).splitlines()
        assert overview[0] == "optimizers" and overview[1].split() == ["goes", "in", "optimizers.<name>"]
        await pilot.press("enter")
        await pilot.pause()
        assert branch.is_expanded
        await pilot.press("down")
        await pilot.pause()
        assert tree.cursor_node.data.name == "adam"
        content = app.query_one("#legos-page").content
        lines = plain(content).splitlines()
        assert lines[0] == "adam" and any(line.split() == ["category", "optimizers"] for line in lines)
        assert any(isinstance(getattr(item, "renderable", None), Syntax) for item in content.renderables)
