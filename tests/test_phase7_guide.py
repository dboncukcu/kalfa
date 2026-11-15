import shutil
from pathlib import Path

import pytest
from rich.console import Console

from kalfa.commands import guide
from kalfa.tui.app import KalfaApp


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "01_mlp_regression" / "config.yaml"
SECTIONS = ["params", "run", "data", "models", "losses", "metrics", "optimizers", "train", "rules", "plots"]


@pytest.fixture(scope="module")
def doc():
    return guide.document([str(EXAMPLE)])


def section(doc, name):
    return next(item for item in doc["sections"] if item["name"] == name)


def entry(doc, path):
    return next(item for part in doc["sections"] for item in part["entries"] if item["path"] == path)


def notes(doc, path):
    return " ".join(entry(doc, path)["notes"])


def test_sections_follow_the_config_and_gather_the_run_values(doc):
    assert doc["ok"] is True and doc["title"].endswith("config.yaml") and doc["problems"] == []
    assert [item["name"] for item in doc["sections"]] == SECTIONS
    assert [item["label"] for item in section(doc, "run")["entries"]] == ["seed", "record"]
    colors = [item["color"] for item in doc["sections"]]
    assert len(set(colors)) == len(colors)
    assert all(item["about"].strip() for item in doc["sections"])


def test_entries_carry_their_yaml_and_the_notes_from_every_source(doc):
    assert [item["label"] for item in section(doc, "losses")["entries"]] == ["mse", "huber", "l1", "log_cosh"]
    assert {item["label"] for item in section(doc, "data")["entries"]} >= {"use", "file", "split", "features",
                                                                         "targets", "batch"}
    net = entry(doc, "models.net")
    assert net["label"] == "net" and net["yaml"].startswith("net:") and "mlp" in net["yaml"]
    assert "hidden" in notes(doc, "models.net")
    assert "net" in notes(doc, "losses.mse") and "price" in notes(doc, "losses.mse")
    main = notes(doc, "optimizers.main")
    assert "net" in main and "mse" in main and "0.001" in main
    assert "train.epochs" in notes(doc, "params.epochs")
    rule = notes(doc, "rules.to_huber")
    assert "50" in rule and "optimizers.main.minimizes" in rule and "huber" in rule
    assert "standard_scaler" in notes(doc, "data.features")
    assert notes(doc, "train.checkpoint") and notes(doc, "plots.loss_curve")


def test_problems_attach_to_their_entry(tmp_path):
    folder = tmp_path / "example"
    shutil.copytree(EXAMPLE.parent, folder, ignore=shutil.ignore_patterns("__pycache__", "*.parquet", "data"))
    config = folder / "config.yaml"
    config.write_text(config.read_text(encoding="utf-8").replace("use: mlp,", "use: mlpp,"), encoding="utf-8")
    found = guide.document([str(config)])
    assert found["ok"] is False
    assert [problem["code"] for problem in entry(found, "models.net")["problems"]] == ["unknown_use"]
    assert entry(found, "models.net")["problems"][0]["level"] == "error"
    missing = entry(found, "data.file")["problems"]
    assert [problem["code"] for problem in missing] == ["data_file_missing"] and missing[0]["level"] == "warning"
    assert {problem["code"] for problem in found["problems"]} >= {"unknown_use", "data_file_missing"}


def test_a_missing_file_gives_a_document_that_says_so(tmp_path):
    found = guide.document([str(tmp_path / "nothing.yaml")])
    assert found["ok"] is False and found["sections"] == []
    assert [problem["code"] for problem in found["problems"]] == ["file_not_found"]


def test_text_and_renderable_show_every_section(doc):
    plain = guide.text(doc)
    assert all(name in plain for name in SECTIONS) and "net:" in plain and "\x1b[" not in plain
    console = Console(record=True, width=110, force_terminal=True, color_system="truecolor")
    console.print(guide.renderable(doc))
    assert "\x1b[" in console.export_text(styles=True, clear=False)
    assert "optimizers" in console.export_text()


@pytest.mark.asyncio
async def test_the_guide_tab_reads_the_config_tab(tmp_path):
    app = KalfaApp(root=str(tmp_path), config=str(EXAMPLE))
    async with app.run_test(size=(140, 50)) as pilot:
        await pilot.press("4")
        await pilot.pause()
        listing = app.query_one("#guide-sections")
        names = [listing.get_option_at_index(index).id for index in range(listing.option_count)]
        assert names == SECTIONS
        assert app.focused is listing


def test_an_import_path_of_a_short_name_gets_the_notes_of_that_name(doc):
    huber = notes(doc, "losses.huber")
    assert "short name huber" in huber and "delta:" in huber
