import json
import threading
import time

import pytest

from helpers import minimal, write_config
from kalfa.cli import main
from kalfa.record import Record

pytestmark = pytest.mark.slow


def test_run_predict_resume_and_collect(workdir, capsys):
    config = minimal()
    config["params"]["epochs"] = 2
    config["params"]["fold"] = 0
    config["record"] = "runs/cv_$fold$"
    path = write_config(workdir / "cfg.yaml", config)
    assert main(["run", path]) == 0
    out = capsys.readouterr().out
    assert "record runs/cv_0" in out and (workdir / "runs" / "cv_0" / "history.jsonl").exists()
    assert main(["run", path, "-p", "fold=1"]) == 0
    capsys.readouterr()
    assert main(["run", path]) == 1
    assert "exists" in capsys.readouterr().err

    assert main(["predict", "runs/cv_0", "--which", "last"]) == 0
    assert "predicted 300 rows" in capsys.readouterr().out
    assert main(["predict", "runs/cv_0", "--data", "housing.parquet"]) == 0
    assert "predictions_housing.parquet" in capsys.readouterr().out

    assert main(["resume", "runs/cv_0", "--set", "training.epochs=3"]) == 1
    assert "exists" in capsys.readouterr().err
    resolved = workdir / "runs" / "cv_0" / "resolved.yaml"
    assert "record: runs/cv_0\n" in resolved.read_text()
    resolved.write_text(resolved.read_text().replace("record: runs/cv_0\n", "record: runs/cv_0_resumed\n"))
    assert main(["resume", "runs/cv_0", "--set", "training.epochs=3"]) == 0
    assert "resumed" in capsys.readouterr().out
    lines = (workdir / "runs" / "cv_0_resumed" / "history.jsonl").read_text().splitlines()
    assert [json.loads(line)["turn"] for line in lines] == [3]

    assert main(["collect", "runs/cv_0", "runs/cv_1"]) == 0
    out = capsys.readouterr().out
    assert "CROSS VALIDATION" in out and (workdir / "runs" / "cv.json").exists()
    assert "# k fold summary" in (workdir / "runs" / "cv.md").read_text()
    summary = json.loads((workdir / "runs" / "cv.json").read_text())
    assert [fold["fold"] for fold in summary["folds"]] == [0, 1] and "test/rmse" in summary["summary"]


def test_run_log_info_narrates_the_run(workdir, capsys):
    path = write_config(workdir / "cfg.yaml", minimal())
    assert main(["run", path, "--log"]) == 0
    err = capsys.readouterr().err
    assert "INFO   data.source" in err and "reading housing.parquet" in err
    assert "2000 rows, 9 columns" in err
    assert "random: train 1400, valid 300, test 300" in err
    assert "9 fields -> 8 features, 1 targets" in err
    assert "train: 11 batches of 128" in err
    assert "net: adam lr 0.001 over net, loss loss_mse" in err
    assert "turn 1" in err and "predictions.parquet: 300 rows" in err
    assert "plots: loss_curve, pred_vs_true" in err and "finished in" in err
    assert "training.epochs" not in err


def test_run_log_debug_adds_the_nodes_and_the_decisions(workdir, capsys):
    path = write_config(workdir / "cfg.yaml", minimal())
    assert main(["run", path, "--log", "debug"]) == 0
    err = capsys.readouterr().err
    assert "DEBUG  data.source" in err and "started" in err
    assert "training.epochs[0].turn" in err
    assert "std_scaler on 8 columns" in err
    assert "applying the chains to the train set" in err
    assert "turn 1: 11 steps" in err and "wrote best.pt, last.pt" in err


def test_run_no_progress_leaves_the_bar_out(workdir, capsys):
    config = minimal()
    config["record"] = "runs/quiet"
    path = write_config(workdir / "cfg.yaml", config)
    assert main(["run", path, "--no-progress"]) == 0
    assert "%|" not in capsys.readouterr().err
    config["record"] = "runs/logged"
    path = write_config(workdir / "logged.yaml", config)
    assert main(["run", path, "--log", "--no-progress"]) == 0
    err = capsys.readouterr().err
    assert "turn 1" in err and "%|" not in err


def test_run_without_log_prints_nothing_extra(workdir, capsys):
    path = write_config(workdir / "cfg.yaml", minimal())
    assert main(["run", path]) == 0
    captured = capsys.readouterr()
    assert "reading housing.parquet" not in captured.err and "INFO" not in captured.err
    assert "reading housing.parquet" not in captured.out


def test_predict_takes_the_log_option(workdir, capsys):
    config = minimal()
    config["record"] = "runs/one"
    path = write_config(workdir / "cfg.yaml", config)
    assert main(["run", path]) == 0
    capsys.readouterr()
    assert main(["predict", "runs/one", "--log", "info"]) == 0
    err = capsys.readouterr().err
    assert "predicting with net (best weights)" in err and "300 rows ->" in err


def test_progress_steps_log_every_and_export(workdir, capsys):
    config = minimal()
    config["record"] = "runs/stepped"
    path = write_config(workdir / "cfg.yaml", config)
    assert main(["run", path, "--log", "--progress", "steps", "--log-every", "5"]) == 0
    err = capsys.readouterr().err
    assert "training.step   step 5 " in err and "step 1 " not in err and "turn 1" in err
    assert (workdir / "runs" / "stepped" / "steps.jsonl").exists()
    assert (workdir / "runs" / "stepped" / "git.json").exists()
    assert main(["export", "runs/stepped", "--format", "pt2", "--which", "final"]) == 0
    out = capsys.readouterr().out
    assert "exported net as /export/kalfa/pt2" in out
    assert (workdir / "runs" / "stepped" / "export" / "net.pt2").exists()
    assert main(["export", "runs/stepped", "--format", "adam"]) == 1


def test_plots_command_and_predict_plots(workdir, capsys):
    config = minimal()
    config["record"] = "runs/drawn"
    path = write_config(workdir / "cfg.yaml", config)
    assert main(["run", path]) == 0
    (workdir / "runs" / "drawn" / "plots" / "loss_curve.png").unlink()
    assert main(["plots", "runs/drawn", "--only", "loss_curve"]) == 0
    assert "plots loss_curve" in capsys.readouterr().out
    assert (workdir / "runs" / "drawn" / "plots" / "loss_curve.png").exists()
    assert main(["predict", "runs/drawn", "--plots", "pred_vs_true"]) == 0
    assert "plots pred_vs_true" in capsys.readouterr().out


def test_a_stop_file_ends_the_run_after_the_turn_it_is_seen_in(workdir, capsys):
    config = minimal()
    config["params"]["epochs"] = 8
    config["record"] = "runs/stopped"
    path = write_config(workdir / "cfg.yaml", config)
    record = workdir / "runs" / "stopped"

    def ask():
        while not (record / "history.jsonl").exists():
            time.sleep(0.02)
        Record(record).request_stop("test")

    thread = threading.Thread(target=ask, daemon=True)
    thread.start()
    assert main(["run", path, "--log", "--no-progress"]) == 0
    thread.join()
    capsys.readouterr()
    turns = len((record / "history.jsonl").read_text().splitlines())
    assert 1 <= turns < 8 and (record / "final" / "state.pt").exists() and (record / "predictions.parquet").exists()
    assert json.loads((record / "run.json").read_text())["status"] == "ok"
    assert "stop requested by test" in (record / "stderr.txt").read_text()
