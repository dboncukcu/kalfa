import json
import re
import statistics
from pathlib import Path

import pytest

from data import write_housing
from helpers import inside, load_config, write_config
from kalfa.cli import main
from kalfa.record import Record


@pytest.fixture(scope="module")
def folds(tmp_path_factory):
    home = tmp_path_factory.mktemp("cv")
    write_housing(home / "housing.parquet")
    config = load_config("tiny")
    config["params"]["fold"] = 0
    config["data"]["split"] = {"uri": "kfold", "params": {"k": 3, "fold": "$fold$", "val": 0.2, "seed": 1}}
    config["record"] = "runs/cv_$fold$"
    path = write_config(home / "cv.yaml", config)
    with inside(home):
        for fold in range(3):
            assert main(["run", path, "-p", f"fold={fold}", "--no-progress"]) == 0
    return home


def sweep_root(tmp_path):
    root = Record(tmp_path / "grid")
    root.manifest("sweep", strategy="/strategy/kalfa/grid", total=4,
                  objective={"monitor": "val/rmse", "mode": "min", "at": "best"})
    for index, (lr, value) in enumerate(((0.1, 0.5), (0.2, 0.4))):
        point = Record(root.directory / f"{index:04d}")
        point.manifest("point", id=index, values={"lr": lr})
        point.write_json("sweep.json", {
            "id": index, "point": {"lr": lr}, "strategy": "/strategy/kalfa/grid", "total": 4,
            "objective": {"monitor": "val/rmse", "mode": "min", "at": "best", "value": value, "turn": 2},
            "record": str(point.directory)})
        point.append("history.jsonl", {"turn": 1, "val/rmse": value + 0.2, "test/rmse": value + 0.3, "rules": []})
        point.append("history.jsonl", {"turn": 2, "val/rmse": value, "test/rmse": value + 0.1, "rules": []})
    Record(root.directory / "0002").write_json("run.json", {"status": "failed"})
    Record(root.directory / "0003").manifest("point", id=3, values={"lr": 0.4})
    return root.directory


@pytest.mark.slow
def test_collect_summarizes_the_folds_with_their_mean_and_deviation(folds, monkeypatch, capsys):
    monkeypatch.chdir(folds)
    assert main(["collect", "runs/cv_0", "runs/cv_1", "runs/cv_2"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("── CROSS VALIDATION ")
    assert "  folds      3: 0, 1, 2\n" in out
    assert re.search(r"^  metric\s+mean\s+std$", out, re.M)
    assert re.search(r"^  test/mse\s+[\d.]+\s+[\d.]+$", out, re.M)
    assert re.search(r"^  test/rmse\s+[\d.]+\s+[\d.]+$", out,
                                                                                  re.M)
    assert (
        out.endswith("  per fold   cv.md carries the value every fold reached\n\nwrote cv.json and cv.md under runs\n"))
    summary = json.loads((folds / "runs" / "cv.json").read_text())
    assert [fold["fold"] for fold in summary["folds"]] == [0, 1, 2]
    assert [fold["dir"] for fold in summary["folds"]] == ["runs/cv_0", "runs/cv_1", "runs/cv_2"]
    assert [fold["turns"] for fold in summary["folds"]] == [1, 1, 1]
    assert list(summary["summary"]) == ["test/mse", "test/rmse"]
    values = [fold["last"]["test/rmse"] for fold in summary["folds"]]
    assert summary["summary"]["test/rmse"] == {"mean": pytest.approx(statistics.mean(values)),
                                               "std": pytest.approx(statistics.stdev(values))}
    report = (folds / "runs" / "cv.md").read_text()
    assert report.startswith("# k fold summary\n\n| metric | mean | std |\n|---|---|---|\n| test/mse | ")
    assert "\n| fold | turns | test/mse | test/rmse |\n|---|---|---|---|\n| 0 | 1 | " in report
    assert re.search(r"^\| 2 \| 1 \| [\d.]+ \| [\d.]+ \|$", report, re.M)


@pytest.mark.slow
def test_collect_markdown_prints_the_report_it_writes(folds, monkeypatch, capsys):
    monkeypatch.chdir(folds)
    assert main(["collect", "runs/cv_0", "runs/cv_1", "runs/cv_2", "--markdown", "--out", "reports"]) == 0
    out = capsys.readouterr().out
    report = (folds / "reports" / "cv.md").read_text()
    assert out == report + "wrote cv.json and cv.md under reports\n"
    assert sorted(path.name for path in (folds / "reports").iterdir()) == ["cv.json", "cv.md"]


@pytest.mark.slow
def test_collect_of_one_fold_run_is_a_summary_of_one(folds, monkeypatch, capsys):
    monkeypatch.chdir(folds)
    assert main(["collect", "runs/cv_1", "--out", "single"]) == 0
    out = capsys.readouterr().out
    assert "  folds      1: 1\n" in out and out.endswith("wrote cv.json and cv.md under single\n")
    summary = json.loads((folds / "single" / "cv.json").read_text())
    assert [fold["fold"] for fold in summary["folds"]] == [1] and summary["summary"]["test/rmse"]["std"] == 0.0
    assert summary["summary"]["test/rmse"]["mean"] == summary["folds"][0]["last"]["test/rmse"]


def test_collect_of_a_run_tabulates_its_last_values(reference, tmp_path, capsys):
    out_dir = tmp_path / "out"
    assert main(["collect", reference.record, "--out", str(out_dir)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("── RUNS ") and re.search(r"^  dir\s+turns\s+test/acc\s+test/auroc\s+", out, re.M)
    assert out.endswith(f"\nwrote sweep.csv, sweep.json and sweep.md under {out_dir}\n")
    table = json.loads((out_dir / "sweep.json").read_text())
    assert table["varying"] == [] and len(table["rows"]) == 1
    row = table["rows"][0]
    last = json.loads((Path(reference.record) / "history.jsonl").read_text().splitlines()[-1])
    assert row["dir"] == reference.record and row["turns"] == 3
    assert row["val/rmse_lin"] == last["val/rmse_lin"] and row["test/rmse_lin"] == last["test/rmse_lin"]
    assert sorted(key for key in row if "/" in key) == sorted(key for key in last
                                                              if key.startswith(("val/", "test/")))
    assert (out_dir / "sweep.md").read_text().startswith("# sweep\n\n| dir | turns | test/acc | test/auroc | ")


def test_collect_of_a_sweep_root_finds_the_best_point_and_the_skipped_ones(tmp_path, capsys):
    root = sweep_root(tmp_path)
    assert main(["collect", str(root)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("── SWEEP ")
    assert "  objective  val/rmse (min, best)\n" in out and "  points     2 of 4 finished\n" in out
    assert re.search(r"^\s+id\s+lr\s+objective\s+turn$", out, re.M)
    assert re.search(r"^\s+0\s+0\.1\s+0\.5\s+2/2$", out, re.M)
    assert re.search(r"^  \*\s+1\s+0\.2\s+0\.4\s+2/2$", out, re.M)
    collapsed = " ".join(out.split())
    assert "best point 1, val/rmse = 0.4 at turn 2," in collapsed
    assert "lr=0.2 failed 1: 0002 unfinished 1: 0003" in collapsed
    assert re.search(r"^  failed\s+1: 0002$", out, re.M) and re.search(r"^  unfinished\s+1: 0003$", out, re.M)
    assert "  metrics    2 per point, in sweep.csv and sweep.md, or here with --markdown\n" in out
    assert out.endswith(f"\nwrote sweep.csv, sweep.json and sweep.md under {root}\n")
    summary = json.loads((root / "sweep.json").read_text())
    assert summary["objective"] == {"monitor": "val/rmse", "mode": "min", "at": "best"}
    assert summary["best"] == {"id": 1, "point": {"lr": 0.2}, "value": 0.4, "turn": 2, "dir": str(root / "0001")}
    assert summary["points"] == [
        {"id": 0, "lr": 0.1, "objective": 0.5, "turn": 2, "turns": 2, "val/rmse": 0.5, "test/rmse": 0.6},
        {"id": 1, "lr": 0.2, "objective": 0.4, "turn": 2, "turns": 2, "val/rmse": 0.4, "test/rmse": 0.5}]
    assert summary["skipped"] == [{"dir": "0002", "status": "failed"}, {"dir": "0003", "status": "unfinished"}]
    assert (root / "sweep.csv").read_text() == ("id,lr,objective,turn,turns,val/rmse,test/rmse\n"
                                                "0,0.1,0.5,2,2,0.5,0.6\n1,0.2,0.4,2,2,0.4,0.5\n")
    reports = tmp_path / "reports"
    assert main(["collect", str(root), "--markdown", "--out", str(reports)]) == 0
    out = capsys.readouterr().out
    report = (reports / "sweep.md").read_text()
    assert out == report + f"wrote sweep.csv, sweep.json and sweep.md under {reports}\n"
    assert report == (
        "# sweep: val/rmse (min, best)\n\n2 of 4 points finished.\n\n"
        "| id | lr | objective | turn | turns | test/rmse | val/rmse |\n|---|---|---|---|---|---|---|\n"
        "| 0 | 0.1 | 0.5 | 2 | 2 | 0.6 | 0.5 |\n| 1 | 0.2 | 0.4 | 2 | 2 | 0.5 | 0.4 |\n\n"
        f"best: point 1 with val/rmse=0.4 at turn 2 ({root / '0001'}), lr=0.2\nfailed: 1 (0002)\nunfinished: 1 "
        f"(0003)\n")
    assert sorted(path.name for path in reports.iterdir()) == ["sweep.csv", "sweep.json", "sweep.md"]


def test_collect_refuses_what_holds_no_run(tmp_path, capsys):
    assert main(["collect", str(tmp_path / "nowhere")]) == 1
    assert capsys.readouterr().err == "no run with a resolved.yaml among the given directories\n"
    empty = Record(tmp_path / "empty")
    empty.manifest("sweep", total=1)
    assert main(["collect", str(empty.directory)]) == 1
    assert capsys.readouterr().err == (f"{empty.directory}: no finished point (a directory with sweep.json) under the "
                                       "sweep root\n")
