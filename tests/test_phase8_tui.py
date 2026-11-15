import json
import sys
from datetime import datetime

import pytest

from kalfa.core.record import RecordWriter
from kalfa.records.reader import Cache, records_under
from kalfa.tui.app import KalfaApp
from kalfa.tui.records import (
    detail_of,
    left_text,
    matches,
    nonfinite_text,
    ordered,
    point_rows,
    read_row,
    record_rows,
    seconds_text,
    summary_text,
    turn_text,
    within_text,
)
from records.fixtures import make_record, make_sweep, write_json


@pytest.fixture
def root(tmp_path):
    make_record(tmp_path / "runs" / "alpha", status="done", params={"lr": 0.001, "width": 64})
    failure = {"type": "RuntimeError", "message": "boom", "traceback": "Traceback\n  File x\nRuntimeError: boom"}
    make_record(tmp_path / "runs" / "beta", status="failed", params={"lr": 0.01, "width": 32}, turns=2,
                failure=failure)
    make_sweep(tmp_path / "runs" / "sweep", count=3)
    return tmp_path / "runs"


def names_of(app):
    table = app.query_one("#records-table")
    return [str(table.get_row_at(index)[0]) for index in range(table.row_count)]


def log_of(app):
    return "\n".join(app.query_one("#records-log").plain)


def test_the_texts_of_the_turn_in_progress_read_short():
    assert seconds_text(42) == "42s" and seconds_text(125) == "2m 05s" and seconds_text(7300) == "2h 01m"
    assert left_text(30) == "under a minute left" and left_text(125.0) == "about 2m 05s left"
    assert left_text(None) == "" and left_text(True) == ""
    within = {"turn": 4, "phase": "train", "done": 50, "total": 200, "unit": "batches"}
    assert within_text(within) == "batch 50 of 200"
    assert within_text({**within, "unit": "updates"}) == "update 50 of 200"
    assert within_text({**within, "phase": "valid", "done": 3, "total": 9}) == "valid batch 3 of 9"
    assert within_text({**within, "total": None}) == "50 batches"
    assert within_text({"phase": "plots", "done": None, "total": None}) == "plots" and within_text(None) == ""
    assert turn_text({"turn": 3, "total": 10, "within": within, "fraction": 0.325}) == "3/10 (32%)"
    assert turn_text({"turn": 3, "total": 10, "fraction": 0.3}) == "3/10"
    assert nonfinite_text({"turn": 2, "series": ["train/mse", "val/mse"], "value": "nan"}) == (
        "train/mse and 1 more is nan from turn 2")
    assert nonfinite_text(None) == ""


def test_the_summary_text_says_where_the_turn_is_and_what_is_not_finite():
    text = summary_text({"name": "x", "status": "running", "turn": 3, "total": 10, "unit": "epochs", "eta": 400.0,
                         "within": {"turn": 4, "phase": "train", "done": 5, "total": 20, "unit": "batches"},
                         "fraction": 0.325, "nonfinite": {"turn": 2, "series": ["val/mse"], "value": "inf"}})
    assert "status    running, turn 3/10 (32%) epochs" in text
    assert "now       batch 5 of 20, about 6m 40s left" in text
    assert "warning   val/mse is inf from turn 2" in text


def test_the_detail_of_a_failed_record_ends_with_its_traceback(root):
    text, _ = detail_of(root, {"path": "beta", "kind": "record"})
    assert "failure   boom" in text and "traceback (its end; log.txt and the board have all of it)" in text
    assert text.rstrip().endswith("RuntimeError: boom")
    done, _ = detail_of(root, {"path": "alpha", "kind": "record"})
    assert "traceback" not in done


def test_rows_carry_what_the_table_sorts_and_filters_by(root):
    rows = record_rows(root)
    assert [row["key"] for row in rows] == ["alpha", "beta", "sweep"]
    alpha, beta, swept = rows
    assert alpha["turn"] == 3 and beta["turn"] == 2 and swept["turn"] == 2
    assert alpha["words"] == "alpha lr=0.001 width=64" and isinstance(alpha["best"], float)
    assert matches(alpha, ["width=64"]) and matches(alpha, ["alpha", "done"]) and not matches(alpha, ["failed"])
    assert matches(beta, ["failed", "lr=0.01"]) and matches(swept, ["sweep"])
    assert [row["key"] for row in ordered(rows, "started")] == ["alpha", "beta", "sweep"]
    assert [row["key"] for row in ordered(rows, "started", True)] == ["sweep", "beta", "alpha"]
    assert [row["key"] for row in ordered(rows, "name", True)] == ["sweep", "beta", "alpha"]
    assert [row["key"] for row in ordered(rows, "status")] == ["alpha", "beta", "sweep"]
    assert [row["key"] for row in ordered(rows, "turn")][-1] == "alpha"
    by_best = ordered(rows + [{**alpha, "key": "none", "best": None}], "best")
    assert by_best[-1]["key"] == "none" and by_best[0]["best"] == min(row["best"] for row in rows)
    assert ordered(rows + [{**alpha, "key": "none", "best": None}], "best", True)[-1]["key"] == "none"


def test_the_rows_of_a_sweep_are_its_points_by_their_path(root):
    cache = Cache()
    rows = point_rows(root, "sweep", cache)
    assert [row["key"] for row in rows] == ["sweep/0000", "sweep/0001", "sweep/0002"]
    assert rows[0]["cells"][0] == "0000  lr=0.01 width=32" and rows[0]["cells"][1] == "done"
    assert rows[0]["cells"][3].startswith("val/rmse ") and rows[2]["cells"][1:] == ("failed", "3/10", "")
    assert rows[0]["entry"] == {"path": "sweep/0000", "kind": "record"} and rows[2]["best"] is None
    assert matches(rows[1], ["width=64"]) and not matches(rows[0], ["width=64"])
    assert [row["key"] for row in point_rows(root / "sweep", ".", cache)] == ["0000", "0001", "0002"]
    assert cache.hits >= 3


def test_a_row_reads_its_record_through_the_cache(root):
    cache = Cache()
    entry = records_under(root)[0]
    first = read_row(root, entry, cache)
    assert read_row(root, entry, cache) == first and (cache.hits, cache.misses) == (1, 1)


@pytest.mark.asyncio
async def test_the_filter_narrows_the_table_and_the_order_key_sorts_it(root):
    app = KalfaApp(root=str(root))
    async with app.run_test() as pilot:
        await pilot.pause()
        assert names_of(app) == ["alpha", "beta", "sweep"]
        assert str(app.query_one("#records-order").content) == "3 of 3 records, by started"
        await pilot.press("slash")
        await pilot.pause()
        assert app.focused is app.query_one("#records-filter")
        await pilot.press(*"failed")
        await pilot.pause()
        assert names_of(app) == ["beta"]
        assert str(app.query_one("#records-order").content) == "1 of 3 records, by started"
        await pilot.press("enter")
        await pilot.pause()
        assert app.focused is app.query_one("#records-table")
        assert str(app.query_one("#record-detail").content).startswith("beta\n")
        app.query_one("#records-filter").value = ""
        await pilot.pause()
        assert names_of(app) == ["alpha", "beta", "sweep"]
        await pilot.press("t")
        await pilot.pause()
        assert str(app.query_one("#records-order").content) == "3 of 3 records, by name"
        await pilot.press("t", "t")
        await pilot.pause()
        assert str(app.query_one("#records-order").content) == "3 of 3 records, by turn"
        assert names_of(app)[-1] == "alpha"


@pytest.mark.asyncio
async def test_enter_lists_the_points_of_a_sweep_and_backspace_comes_back(root, monkeypatch):
    launched = []
    monkeypatch.setattr(KalfaApp, "launch", lambda self, argv, label, target, detach=True: launched.append(argv))
    app = KalfaApp(root=str(root))
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#records-table").move_cursor(row=2)
        await pilot.pause()
        assert "enter lists the points" in str(app.query_one("#record-detail").content)
        await pilot.press("enter")
        await pilot.pause()
        assert [name.split()[0] for name in names_of(app)] == ["0000", "0001", "0002"]
        assert str(app.query_one("#records-order").content).startswith("3 of 3 points of sweep (backspace goes back)")
        assert app.sub_title == str(root / "sweep")
        assert str(app.query_one("#record-detail").content).startswith("0000\n")
        await pilot.press("p")
        await pilot.pause()
        assert launched == [[sys.executable, "-m", "kalfa.cli", "predict", str(root / "sweep" / "0000")]]
        await pilot.press("r")
        await pilot.pause()
        assert len(names_of(app)) == 3
        await pilot.press("backspace")
        await pilot.pause()
        assert names_of(app) == ["alpha", "beta", "sweep"] and app.sub_title == str(root)
        assert app.query_one("#records-table").cursor_row == 2


@pytest.mark.asyncio
async def test_the_u_key_resumes_a_record_that_does_not_run(root, monkeypatch):
    launched = []
    monkeypatch.setattr(KalfaApp, "launch",
                        lambda self, argv, label, target, detach=True: launched.append((argv, label)))
    manifest = json.loads((root / "alpha" / "manifest.json").read_text(encoding="utf-8"))
    write_json(root / "alpha" / "manifest.json", {**manifest, "status": "running", "finished": None,
                                                  "updated": datetime.now().isoformat(timespec="seconds")})
    app = KalfaApp(root=str(root))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("u")
        await pilot.pause()
        assert launched == [] and "is running; s asks it to stop after its turn" in log_of(app)
        await pilot.press("down", "u")
        await pilot.pause()
        assert launched == [([sys.executable, "-m", "kalfa.cli", "run", "--from", str(root / "beta")], "resume")]
        await pilot.press("down", "u")
        await pilot.pause()
        assert len(launched) == 1 and "resume works on a record and sweep is a sweep root" in log_of(app)


@pytest.mark.asyncio
async def test_the_l_key_prints_the_end_of_the_log(root):
    writer = RecordWriter(root / "alpha")
    for index in range(60):
        writer.log(f"line {index}")
    app = KalfaApp(root=str(root))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("l")
        await pilot.pause()
        text = log_of(app)
        assert f"log: the last 40 lines of {root / 'alpha'}/log.txt" in text
        assert "line 59" in text and "line 20" in text and "line 19\n" not in text + "\n"
        await pilot.press("down", "l")
        await pilot.pause()
        assert f"log: {root / 'beta'} has no log.txt" in log_of(app)


@pytest.mark.asyncio
async def test_the_timer_reads_the_records_aside_and_keeps_the_cursor(root):
    app = KalfaApp(root=str(root))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("down")
        await pilot.pause()
        make_record(root / "gamma", started="2026-11-08T10:00:00")
        view = app.query_one("#records-view")
        worker = view.reload_aside()
        await worker.wait()
        await pilot.pause()
        assert names_of(app) == ["gamma", "alpha", "beta", "sweep"]
        assert app.query_one("#records-table").cursor_row == 2
        assert view.cache.hits >= 2
