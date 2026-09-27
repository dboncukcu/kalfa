import json
import math

import numpy
import pandas
import pytest

from kalfa.board import Board, report_argv
from kalfa.cli import main
from kalfa.record import Record, write_failure


def point_lines(values):
    return [{"turn": turn, "global_step": turn, "train/loss": train, "val/loss": loss, "val/rmse": rmse,
             "test/rmse": test, "test/acc": acc, "seconds": seconds, "rules": []}
            for turn, (train, loss, rmse, test, acc, seconds) in enumerate(values, start=1)]


def labelled_sweep(tmp_path):
    root = Record(tmp_path / "grid")
    root.manifest("sweep", strategy="/strategy/kalfa/grid", total=4,
                  objective={"monitor": "val/rmse", "mode": "min", "at": "best"},
                  space={"lr": "Choices(values=[0.1, 0.2, 0.4, 0.8])"})
    y = numpy.arange(1, 481, dtype="float64")
    hot = (numpy.arange(480) % 2).astype("float64")
    shapes = {0: (y * 1.1, numpy.zeros(480), numpy.full(480, 0.5)),
              1: (y.copy(), hot.copy(), hot * 0.8 + 0.1)}
    histories = {0: [(1.2, 1.8, 3.0, 3.1, 0.5, 10.0), (1.0, 1.5, 2.0, 2.2, 0.6, 10.0),
                     (0.9, 1.6, 2.5, 2.6, 0.55, 10.0)],
                 1: [(0.8, 1.0, 2.0, 2.1, 0.8, 12.0), (0.5, 0.7, 1.0, 1.1, 0.9, 12.0),
                     (0.4, 0.8, 1.5, 1.6, 0.85, 12.0)]}
    for index, lr in enumerate((0.1, 0.2, 0.4, 0.8)):
        point = Record(root.directory / f"{index:04d}")
        point.manifest("point", id=index, values={"lr": lr}, root=str(root.directory), turn="epoch")
        point.write_text("resolved.yaml", f"params:\n  lr: {lr}\nseed: 7\n")
        if index in histories:
            for line in point_lines(histories[index]):
                point.append("history.jsonl", line)
            value = min(line[2] for line in histories[index])
            point.write_json("sweep.json", {"id": index, "point": {"lr": lr}, "objective": {
                "monitor": "val/rmse", "mode": "min", "at": "best", "value": value, "turn": 2}})
            point.write_json("run.json", {"status": "ok"})
            guess, predicted, score = shapes[index]
            pandas.DataFrame({"row": numpy.arange(480), "y": y, "pred_head_y": guess, "is_hot": hot,
                              "pred_tail_is_hot": predicted, "raw_tail": score}).to_parquet(
                point.path("predictions.parquet"))
    write_failure(root.directory / "0002", RuntimeError("the loader died"))
    return root.directory


def test_report_writes_every_section_of_the_chosen_points(tmp_path, capsys):
    root = labelled_sweep(tmp_path)
    assert main(["report", str(root), "--points", "0,1,2", "--task", "is_hot=classification",
                 "--score", "is_hot=raw_tail", "--signal", "is_hot=1"]) == 0
    written = root / "reports" / "report"
    out = capsys.readouterr().out
    assert out.startswith("wrote report.html with ") and out.endswith(f" under {written}\n")
    assert "points.csv, metrics.csv, regression_y_pred_head_y.csv and classification_is_hot_pred_tail_is_hot.csv" in out
    summary = json.loads((written / "report.json").read_text())
    assert summary["points"] == ["point 0", "point 1", "point 2"]
    assert sorted(summary["files"]) == sorted(["points.csv", "metrics.csv", "regression_y_pred_head_y.csv",
                                               "classification_is_hot_pred_tail_is_hot.csv", "report.html",
                                               "report.json"])
    wins = summary["standings"]["wins"]
    assert wins["point 1"] == {"min": ["test/rmse", "val/loss", "val/rmse"], "max": ["test/acc"]}
    assert wins["point 0"] == {"min": ["test/acc"], "max": ["test/rmse", "val/loss", "val/rmse"]}
    assert wins["point 2"] == {"min": [], "max": []}
    rmse = next(row for row in summary["standings"]["metrics"] if row["metric"] == "val/rmse")
    assert rmse["min"] == 1.0 and rmse["min at"] == ["point 1"] and rmse["max"] == 2.0 and rmse["max at"] == ["point 0"]
    assert rmse["gap to the next above"] == 1.0 and rmse["spread over the min"] == 1.0
    training = {row["point"]: row for row in summary["training"]}
    assert training["point 0"]["turns"] == 3 and training["point 0"]["objective turn"] == 2
    assert training["point 0"]["turns after it"] == 1 and training["point 0"]["seconds"] == 30.0
    assert training["point 0"]["gaps"] == {"loss": pytest.approx(0.5)} and training["point 2"]["turns"] == 0
    regression, classification = summary["targets"]
    assert regression["task"] == "regression" and regression["target"] == "y"
    rows = {row["point"]: row for row in regression["points"]}
    assert rows["point 1"]["rmse"] == 0.0 and rows["point 1"]["response"] == 1.0 and rows["point 1"]["within 10 %"] == 1
    assert rows["point 0"]["response"] == pytest.approx(1.1) and rows["point 0"]["resolution"] == pytest.approx(0.0)
    assert rows["point 0"]["rmse"] == pytest.approx(math.sqrt(numpy.mean((numpy.arange(1, 481) * 0.1) ** 2)))
    assert [len(regression["response"][label]) for label in ("point 0", "point 1")] == [12, 12]
    assert classification["task"] == "classification" and classification["signal"] == "1"
    found = {row["point"]: row for row in classification["points"]}
    assert found["point 1"]["accuracy"] == 1.0 and found["point 1"]["auc"] == 1.0
    assert found["point 0"]["accuracy"] == 0.5 and found["point 0"]["auc"] == 0.5
    assert found["point 1"]["confusion"] == [[240, 0], [0, 240]]
    page = (written / "report.html").read_text()
    for title in ("Summary", "Every metric at the objective turn", "Who takes what", "Training", "Params",
                  "y as a regression (pred_head_y)", "is_hot as a classification (pred_tail_is_hot)",
                  "Failed and lost", "What the numbers mean"):
        assert f">{title}</h2>" in page
    assert "the loader died" in page and page.count('src="data:image/png;base64,') >= 15
    assert pandas.read_csv(written / "metrics.csv")["point"].tolist() == ["point 0", "point 1", "point 2"]


def test_report_writes_over_the_previous_one(tmp_path, capsys):
    root = labelled_sweep(tmp_path)
    assert main(["report", str(root), "--points", "all", "--task", "is_hot=classification"]) == 0
    written = root / "reports" / "report"
    assert (written / "classification_is_hot_pred_tail_is_hot.csv").exists()
    assert "Give --score is_hot=COLUMN and --signal is_hot=CLASS" in (written / "report.html").read_text()
    assert main(["report", str(root), "--points", "1"]) == 0
    capsys.readouterr()
    assert not (written / "classification_is_hot_pred_tail_is_hot.csv").exists()
    assert sorted(path.name for path in written.iterdir()) == ["metrics.csv", "points.csv",
                                                              "regression_is_hot_pred_tail_is_hot.csv",
                                                              "regression_y_pred_head_y.csv", "report.html",
                                                              "report.json"]
    assert json.loads((written / "report.json").read_text())["points"] == ["point 1"]


def test_report_of_runs_and_what_it_refuses(tmp_path, capsys):
    root = labelled_sweep(tmp_path)
    out = tmp_path / "pair"
    assert main(["report", str(root / "0000"), str(root / "0001"), "--out", str(out)]) == 0
    capsys.readouterr()
    assert json.loads((out / "report.json").read_text())["points"] == ["0000", "0001"]
    assert main(["report", str(root)]) == 1
    assert capsys.readouterr().err == (f"{root} is a sweep root; name its points with --points 3,7,12 or "
                                       f"--points all\n")
    assert main(["report", str(root), "--points", "1,9"]) == 1
    assert capsys.readouterr().err == f"{root}: no point 9 under the root, which holds 4 points\n"
    assert main(["report", str(root), "--points", "1", "--task", "y=ranking"]) == 1
    assert capsys.readouterr().err == "--task takes regression or classification, got y=ranking\n"
    assert main(["report", str(root / "0000"), "--points", "1"]) == 1
    assert capsys.readouterr().err == ("--points picks the points of one sweep root; give record directories "
                                       "without it\n")
    assert main(["report", str(root), "--points", "one"]) == 1
    assert capsys.readouterr().err == "--points takes point ids like 3,7,12, or all; got 'one'\n"


@pytest.mark.subprocess
def test_the_board_runs_the_report_command_as_a_process(tmp_path):
    root = labelled_sweep(tmp_path)
    board = Board(tmp_path)
    argv = report_argv(board.root / "grid", "0,1", ["is_hot=classification"], ["is_hot=raw_tail"], ["is_hot=1"])
    board.reports.start(board.root / "grid", argv).join(120)
    found = board.report_status("grid")
    assert found["state"] == "done" and found["message"].startswith("wrote report.html with ")
    assert found["file"] == "grid/reports/report/report.html"
    assert json.loads((root / "reports" / "report" / "report.json").read_text())["points"] == ["point 0", "point 1"]
