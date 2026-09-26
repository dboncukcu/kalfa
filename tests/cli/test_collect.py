import json
import re
import statistics
from pathlib import Path

import pandas
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
    assert main(["collect", "runs/cv_0", "runs/cv_1", "runs/cv_2", "--no-figures"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("── CROSS VALIDATION ")
    assert "  folds      3: 0, 1, 2\n" in out
    assert re.search(r"^  metric\s+mean\s+std$", out, re.M)
    assert re.search(r"^  test/mse\s+[\d.]+\s+[\d.]+$", out, re.M)
    assert re.search(r"^  test/rmse\s+[\d.]+\s+[\d.]+$", out,
                                                                                  re.M)
    assert out.endswith("  per fold   cv.md carries the value every fold reached\n\nwrote cv.json and cv.md under "
                        "runs/reports\n")
    summary = json.loads((folds / "runs" / "reports" / "cv.json").read_text())
    assert [fold["fold"] for fold in summary["folds"]] == [0, 1, 2]
    assert [fold["dir"] for fold in summary["folds"]] == ["runs/cv_0", "runs/cv_1", "runs/cv_2"]
    assert [fold["turns"] for fold in summary["folds"]] == [1, 1, 1]
    assert [fold["state"] for fold in summary["folds"]] == ["finished", "finished", "finished"]
    assert list(summary["summary"]) == ["test/mse", "test/rmse"]
    values = [fold["last"]["test/rmse"] for fold in summary["folds"]]
    assert summary["summary"]["test/rmse"] == {"mean": pytest.approx(statistics.mean(values)),
                                               "std": pytest.approx(statistics.stdev(values))}
    report = (folds / "runs" / "reports" / "cv.md").read_text()
    assert report.startswith("# k fold summary\n\n| metric | mean | std |\n|---|---|---|\n| test/mse | ")
    assert "\n| fold | turns | test/mse | test/rmse |\n|---|---|---|---|\n| 0 | 1 | " in report
    assert re.search(r"^\| 2 \| 1 \| [\d.]+ \| [\d.]+ \|$", report, re.M)
    assert ("\n## Folds\n\n| fold | state | turns | record |\n|---|---|---|---|\n"
            "| 0 | finished | 1 | [cv_0](../cv_0) |\n" in report)
    assert "\n## At the reported turn\n" in report and set(summary["reported"]) >= {"val/rmse", "test/rmse"}
    assert main(["collect", "runs/cv_0", "runs/cv_1", "runs/cv_2"]) == 0
    assert capsys.readouterr().out.endswith("wrote cv.json and cv.md with 1 figure under runs/reports\n")
    assert (folds / "runs" / "reports" / "plots" / "curves.png").exists()
    assert "\n## Curves\n\n![val/rmse over the epochs, one line per fold](plots/curves.png)\n" in (
        folds / "runs" / "reports" / "cv.md").read_text()


@pytest.mark.slow
def test_collect_markdown_prints_the_report_it_writes(folds, monkeypatch, capsys):
    monkeypatch.chdir(folds)
    assert main(["collect", "runs/cv_0", "runs/cv_1", "runs/cv_2", "--markdown", "--out", "reports",
                 "--no-figures"]) == 0
    out = capsys.readouterr().out
    report = (folds / "reports" / "cv.md").read_text()
    assert out == report + "wrote cv.json and cv.md under reports\n"
    assert sorted(path.name for path in (folds / "reports").iterdir()) == ["cv.json", "cv.md"]


@pytest.mark.slow
def test_collect_of_one_fold_run_is_a_summary_of_one(folds, monkeypatch, capsys):
    monkeypatch.chdir(folds)
    assert main(["collect", "runs/cv_1", "--out", "single", "--no-figures"]) == 0
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
    written = root / "reports"
    assert out.endswith(f"\nwrote sweep.csv, sweep.json and sweep.md with 2 figures under {written}\n")
    summary = json.loads((written / "sweep.json").read_text())
    assert summary["objective"] == {"monitor": "val/rmse", "mode": "min", "at": "best"}
    assert summary["best"] == {"id": 1, "point": {"lr": 0.2}, "value": 0.4, "turn": 2, "dir": str(root / "0001")}
    assert summary["points"] == [
        {"id": 0, "lr": 0.1, "objective": 0.5, "turn": 2, "turns": 2, "val/rmse": 0.5, "test/rmse": 0.6},
        {"id": 1, "lr": 0.2, "objective": 0.4, "turn": 2, "turns": 2, "val/rmse": 0.4, "test/rmse": 0.5}]
    assert summary["skipped"] == [{"dir": "0002", "status": "failed", "reason": None},
                                  {"dir": "0003", "status": "unfinished", "reason": None}]
    assert summary["planned"] == 4 and summary["strategy"] == "/strategy/kalfa/grid"
    assert (written / "sweep.csv").read_text() == ("id,lr,objective,turn,turns,val/rmse,test/rmse\n"
                                                   "0,0.1,0.5,2,2,0.5,0.6\n1,0.2,0.4,2,2,0.4,0.5\n")
    assert sorted(path.name for path in (written / "plots").iterdir()) == ["curves.png", "param_lr.png"]
    drawn = (written / "sweep.md").read_text()
    assert "\n## Curves\n\n![val/rmse over the turns](plots/curves.png)\n" in drawn
    assert "\n## Parameters\n\n![val/rmse against lr](plots/param_lr.png)\n" in drawn
    assert "Point 1 ([0001](../0001)) reaches val/rmse = 0.4 at turn 2 of 2.\n" in drawn
    reports = tmp_path / "reports"
    assert main(["collect", str(root), "--markdown", "--out", str(reports), "--no-figures"]) == 0
    out = capsys.readouterr().out
    report = (reports / "sweep.md").read_text()
    assert out == report + f"wrote sweep.csv, sweep.json and sweep.md under {reports}\n"
    assert report.startswith("# sweep: val/rmse (min, best)\n\n2 of 4 points finished, 1 failed, 1 unfinished.\n\n"
                             "Strategy /strategy/kalfa/grid.\n\n## Best point\n\n"
                             "Point 1 ([0001](../grid/0001)) reaches val/rmse = 0.4 at turn 2 of 2.\n\n"
                             "| param | value |\n|---|---|\n| lr | 0.2 |\n\n"
                             "| metric at turn 2 | value |\n|---|---|\n| test/rmse | 0.5 |\n| val/rmse | 0.4 |\n")
    assert ("\n## Top 2\n\n| rank | id | lr | objective | gap to the best | turn | turns |\n"
            "|---|---|---|---|---|---|---|\n| 1 | 1 | 0.2 | 0.4 | 0 | 2 | 2 |\n| 2 | 0 | 0.1 | 0.5 | 0.1 | 2 | 2 |\n"
            in report)
    assert ("\n## Failed and unfinished\n\n| point | status | reason |\n|---|---|---|\n| 0002 | failed |  |\n"
            "| 0003 | unfinished |  |\n" in report)
    assert report.endswith("<details><summary>Every finished point (2)</summary>\n\n"
                           "| id | lr | objective | turn | turns | test/rmse | val/rmse |\n"
                           "|---|---|---|---|---|---|---|\n"
                           "| 0 | 0.1 | 0.5 | 2 | 2 | 0.6 | 0.5 |\n| 1 | 0.2 | 0.4 | 2 | 2 | 0.5 | 0.4 |\n\n"
                           "</details>\n")
    assert "## Parameters" not in report and "## Curves" not in report
    assert sorted(path.name for path in reports.iterdir()) == ["sweep.csv", "sweep.json", "sweep.md"]


def test_collect_refuses_what_holds_no_run(tmp_path, capsys):
    assert main(["collect", str(tmp_path / "nowhere")]) == 1
    assert capsys.readouterr().err == "no run with a resolved.yaml among the given directories\n"
    empty = Record(tmp_path / "empty")
    empty.manifest("sweep", total=1)
    assert main(["collect", str(empty.directory)]) == 1
    assert capsys.readouterr().err == (f"{empty.directory}: no finished point (a directory with sweep.json) under the "
                                       "sweep root\n")


def fold_sweep_root(tmp_path):
    root = Record(tmp_path / "grid")
    root.manifest("sweep", strategy="/strategy/kalfa/grid", total=6,
                  objective={"monitor": "val/rmse", "mode": "min", "at": "best"},
                  space={"lr": "Choices(values=[0.1, 0.2, 0.4])", "fold": "Choices(values=[0, 1])"})
    finished = ((0.1, 0, 0.25), (0.1, 1, 0.75), (0.2, 0, 0.375), (0.2, 1, 0.5), (0.4, 0, 0.125))
    for index, (lr, fold, value) in enumerate(finished):
        point = Record(root.directory / f"{index:04d}")
        point.manifest("point", id=index, values={"lr": lr, "fold": fold})
        point.write_json("sweep.json", {
            "id": index, "point": {"lr": lr, "fold": fold}, "strategy": "/strategy/kalfa/grid", "total": 6,
            "objective": {"monitor": "val/rmse", "mode": "min", "at": "best", "value": value, "turn": 2},
            "record": str(point.directory)})
        point.append("history.jsonl", {"turn": 1, "val/rmse": value + 0.25, "test/rmse": value + 0.375, "rules": []})
        point.append("history.jsonl", {"turn": 2, "val/rmse": value, "test/rmse": value + 0.125, "rules": []})
        point.path("resolved.yaml").write_text(f"params:\n  lr: {lr}\n  fold: {fold}\n")
    Record(root.directory / "0005").write_json("run.json", {"status": "failed"})
    return root.directory


def test_collect_mean_over_ranks_the_settings_by_their_mean_over_the_folds(tmp_path, capsys):
    root = fold_sweep_root(tmp_path)
    reports = tmp_path / "reports"
    assert main(["collect", str(root), "--mean-over", "fold", "--out", str(reports), "--no-figures"]) == 0
    out = capsys.readouterr().out
    assert "── MEAN OVER FOLD " in out and "  settings   2 of 3 with every value of fold\n" in out
    assert re.search(r"^\s+lr\s+mean\s+std\s+values$", out, re.M)
    assert re.search(r"^\s+0\.2\s+0\.4375\s+0\.0883883\s+2$", out, re.M)
    assert re.search(r"^\s+0\.1\s+0\.5\s+0\.353553\s+2$", out, re.M)
    assert "best lr=0.2, mean val/rmse = 0.4375, std 0.0883883" in " ".join(out.split())
    assert out.endswith(f"\nwrote sweep.csv, sweep.json, sweep.md and groups.csv under {reports}\n")
    groups = json.loads((reports / "sweep.json").read_text())["groups"]
    assert groups["over"] == "fold" and groups["levels"] == [0, 1] and groups["keys"] == ["lr"]
    ranked = groups["ranked"]
    assert [item["setting"] for item in ranked] == [{"lr": 0.2}, {"lr": 0.1}]
    assert ranked[0]["mean"] == pytest.approx(0.4375)
    assert ranked[0]["std"] == pytest.approx(statistics.stdev([0.375, 0.5]))
    assert ranked[0]["count"] == 2 and ranked[0]["missing"] == [] and ranked[0]["min"] == 0.375
    assert ranked[0]["points"] == [
        {"id": 2, "level": 0, "value": 0.375, "turn": 2, "dir": str(root / "0002")},
        {"id": 3, "level": 1, "value": 0.5, "turn": 2, "dir": str(root / "0003")}]
    assert ranked[0]["metrics"]["test/rmse"]["mean"] == pytest.approx(0.5625)
    incomplete = [(item["setting"], item["missing"], item["count"]) for item in groups["incomplete"]]
    assert incomplete == [({"lr": 0.4}, [1], 1)]
    table = pandas.read_csv(reports / "groups.csv", keep_default_na=False)
    assert list(table.columns) == ["lr", "mean", "std", "min", "max", "values", "missing", "test/rmse mean",
                                   "test/rmse std", "val/rmse mean", "val/rmse std"]
    assert table["lr"].tolist() == [0.2, 0.1, 0.4] and table["values"].tolist() == [2, 2, 1]
    assert table["missing"].astype(str).tolist() == ["", "", "1"]
    assert table["mean"].tolist() == pytest.approx([0.4375, 0.5, 0.125])
    assert main(["collect", str(root), "--mean-over", "fold", "--out", str(reports), "--no-figures",
                 "--markdown"]) == 0
    report = (reports / "sweep.md").read_text()
    assert capsys.readouterr().out == report + f"wrote sweep.csv, sweep.json, sweep.md and groups.csv under {reports}\n"
    assert report.startswith("# sweep: val/rmse (min, best)\n\n5 of 6 points finished, 1 failed.\n\n"
                             "Strategy /strategy/kalfa/grid.\n\n## Best setting, the mean over fold\n\n"
                             "lr=0.2 reaches a mean val/rmse of 0.4375 with a standard deviation of 0.0883883 over "
                             "the 2 values of fold.\n\n| fold | point | objective | turn |\n|---|---|---|---|\n"
                             "| 0 | [0002](../grid/0002) | 0.375 | 2 |\n| 1 | [0003](../grid/0003) | 0.5 | 2 |\n\n"
                             "| metric at the objective turn | mean | std |\n|---|---|---|\n"
                             "| test/rmse | 0.5625 | 0.0883883 |\n| val/rmse | 0.4375 | 0.0883883 |\n\n"
                             "## Top 2 settings\n\n| rank | lr | mean | std | values | gap to the best |\n"
                             "|---|---|---|---|---|---|\n| 1 | 0.2 | 0.4375 | 0.0883883 | 2 | 0 |\n"
                             "| 2 | 0.1 | 0.5 | 0.353553 | 2 | 0.0625 |\n\n"
                             "## Settings without every value of fold\n\nA setting is ranked once every value of fold "
                             "has a finished point with a finite objective.\n\n| lr | finished | missing |\n"
                             "|---|---|---|\n| 0.4 | 0 | 1 |\n\n## Best point\n\n"
                             "Point 4 ([0004](../grid/0004)) reaches val/rmse = 0.125 at turn 2 of 2.\n")
    assert ("\n## Parameters\n\n| lr | settings | best | median |\n|---|---|---|---|\n| 0.1 | 1 | 0.5 | 0.5 |\n"
            "| 0.2 | 1 | 0.4375 | 0.4375 |\n| 0.4 | 0 |  |  |\n" in report)
    assert ("\n## Config, the best setting against the runner-up at fold = 0\n\n```diff\n--- point 2\n+++ point 0\n"
            in report) and "\n-  lr: 0.2\n+  lr: 0.1\n" in report


def test_collect_mean_over_draws_the_setting_means_and_the_folds_of_the_best_setting(tmp_path, capsys):
    root = fold_sweep_root(tmp_path)
    drawn = tmp_path / "drawn"
    assert main(["collect", str(root), "--mean-over", "fold", "--out", str(drawn)]) == 0
    assert capsys.readouterr().out.endswith(f"wrote sweep.csv, sweep.json, sweep.md and groups.csv with 2 figures "
                                            f"under {drawn}\n")
    assert sorted(path.name for path in (drawn / "plots").iterdir()) == ["curves.png", "param_lr.png"]
    report = (drawn / "sweep.md").read_text()
    assert "\n## Parameters\n\n![mean val/rmse over fold against lr](plots/param_lr.png)\n" in report
    assert "\n## Curves\n\n![val/rmse over the turns, the best setting in colour](plots/curves.png)\n" in report


def test_collect_mean_over_refuses_what_it_cannot_average(tmp_path, capsys):
    root = fold_sweep_root(tmp_path)
    assert main(["collect", str(root), "--mean-over", "width", "--no-figures"]) == 1
    assert capsys.readouterr().err == f"{root}: width is no swept param; the swept params are lr and fold\n"
    plain = sweep_root(tmp_path / "plain")
    assert main(["collect", str(plain), "--mean-over", "lr", "--no-figures"]) == 1
    assert capsys.readouterr().err == (f"{plain}: the space gives lr no list of values; --mean-over averages over the "
                                       "values the space lists\n")
    assert main(["collect", str(root / "0000"), str(root / "0001"), "--mean-over", "fold"]) == 1
    assert capsys.readouterr().err == "--mean-over takes one sweep root (a directory whose manifest says sweep)\n"


def test_collect_rates_every_param_by_its_rank_correlation_or_the_share_its_levels_explain(tmp_path, capsys):
    root = Record(tmp_path / "space")
    root.manifest("sweep", strategy="/strategy/kalfa/random", total=6,
                  objective={"monitor": "val/rmse", "mode": "min", "at": "best"},
                  space={"lr": "Range(low=0.1, high=0.6, log=False)", "width": "Choices(values=[16, 32])"})
    values = ((0.1, 16, 0.9), (0.2, 32, 0.7), (0.3, 16, 0.6), (0.4, 32, 0.4), (0.5, 16, 0.3), (0.6, 32, 0.1))
    for index, (lr, width, value) in enumerate(values):
        point = Record(root.directory / f"{index:04d}")
        point.manifest("point", id=index, values={"lr": lr, "width": width})
        point.write_json("sweep.json", {
            "id": index, "point": {"lr": lr, "width": width}, "strategy": "/strategy/kalfa/random", "total": 6,
            "objective": {"monitor": "val/rmse", "mode": "min", "at": "best", "value": value, "turn": 1},
            "record": str(point.directory)})
        point.append("history.jsonl", {"turn": 1, "val/rmse": value, "rules": []})
    reports = tmp_path / "reports"
    assert main(["collect", str(root.directory), "--out", str(reports), "--no-figures"]) == 0
    capsys.readouterr()
    rated = json.loads((reports / "sweep.json").read_text())["importance"]
    assert [(row["param"], row["measure"], row["direction"], row["count"]) for row in rated] == [
        ("lr", "Spearman ρ", "better as it grows", 6), ("width", "η²", "best mean at 32", 6)]
    assert rated[0]["value"] == pytest.approx(-1.0) and rated[1]["value"] == pytest.approx(0.06 / 0.42)
    report = (reports / "sweep.md").read_text()
    assert ("\n## Parameters\n\n| param | measure | value | direction | points |\n|---|---|---|---|---|\n"
            "| lr | Spearman ρ | -1 | better as it grows | 6 |\n| width | η² | 0.142857 | best mean at 32 | 6 |\n\n"
            "ρ is the Spearman rank correlation of a numeric param with the objective" in report)
