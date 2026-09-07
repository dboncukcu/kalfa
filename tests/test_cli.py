"""The command line: check, run, predict, resume, collect, ls."""

import json
import re

import pytest

import kalfa  # noqa: F401
from helpers import minimal, write_config
from kalfa.cli import main
from kalfa.collect import collect, fold_summary, sweep_table


def test_check_reports_problems_and_exit_codes(workdir, capsys):
    path = write_config(workdir / "cfg.yaml", minimal())
    assert main(["check", path]) == 0
    out = capsys.readouterr().out
    assert "no problems found" in out and "sets (before filters, from the file header): train 1400, valid 300, test 300" in out
    assert "implicit bindings:" in out and "training.init: device <- device" in out
    config = minimal()
    config["training"]["report"] = "best"
    path = write_config(workdir / "bad.yaml", config)
    assert main(["check", path]) == 1
    out = capsys.readouterr().out
    assert "[report_mismatch]" in out and "bad.yaml:" in out


def test_check_layers_prints_the_tree(workdir, capsys):
    base = write_config(workdir / "base.yaml", minimal())
    top = (workdir / "top.yaml")
    top.write_text(f"include: [base.yaml]\ntraining:\n  epochs: 2\n")
    assert main(["check", str(top), "--layers", "-p", "lr=0.5", "--set", "training.epochs=3"]) == 0
    out = capsys.readouterr().out
    assert "layers, bottom to top:" in out and "base.yaml" in out and "--set" in out
    assert "overrides training.epochs" in out


def test_run_predict_resume_collect_ls(workdir, capsys):
    config = minimal()
    config["training"]["epochs"] = 2
    config["params"] = {"fold": 0}
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
    err = capsys.readouterr().err
    assert "exists" in err
    resolved = workdir / "runs" / "cv_0" / "resolved.yaml"
    assert "record: runs/cv_0\n" in resolved.read_text()
    resolved.write_text(resolved.read_text().replace("record: runs/cv_0\n", "record: runs/cv_0_resumed\n"))
    assert main(["resume", "runs/cv_0", "--set", "training.epochs=3"]) == 0
    assert "resumed" in capsys.readouterr().out
    lines = (workdir / "runs" / "cv_0_resumed" / "history.jsonl").read_text().splitlines()
    assert [json.loads(line)["turn"] for line in lines] == [3]

    assert main(["collect", "runs/cv_0", "runs/cv_1"]) == 0
    out = capsys.readouterr().out
    assert "k fold summary" in out and (workdir / "runs" / "cv.json").exists()
    summary = json.loads((workdir / "runs" / "cv.json").read_text())
    assert [fold["fold"] for fold in summary["folds"]] == [0, 1] and "test/rmse" in summary["summary"]

    assert main(["check", "cfg.yaml", "--load"]) == 0
    printed = capsys.readouterr().out
    assert "sets (before filters, from the file header): train" in printed and "sets (loaded): train" in printed
    assert main(["ls", "/alias/kalfa/tabular"]) == 0
    out = capsys.readouterr().out
    assert "/alias/kalfa/tabular" in out and re.search(r"parquet\s+source\s+/source/kalfa/parquet", out)
    assert main(["ls", "/criterion", "--kind", "criterion"]) == 0
    out = capsys.readouterr().out
    assert "/criterion/kalfa/mse" in out and "partial: True" in out
    assert main(["ls"]) == 0
    out = capsys.readouterr().out
    assert "/alias/kalfa/tabular" in out and "/turn/kalfa/alternating" in out


def test_collect_sweep_table(tmp_path):
    for name, lr in (("a", 0.1), ("b", 0.2)):
        run_dir = tmp_path / name
        run_dir.mkdir()
        (run_dir / "resolved.yaml").write_text(f"params:\n  lr: {lr}\n  seed: 1\nrecord: x\n")
        (run_dir / "history.jsonl").write_text(json.dumps({"turn": 1, "val/rmse": lr * 2, "test/rmse": lr}) + "\n")
    kind, text, target = collect([str(tmp_path / "a"), str(tmp_path / "b")])
    assert kind == "sweep" and "| lr |" in text.replace("| dir | turns | lr |", "| lr |") and target == str(tmp_path)
    table = json.loads((tmp_path / "sweep.json").read_text())
    assert table["varying"] == ["lr"] and table["rows"][1]["val/rmse"] == 0.4
    with pytest.raises(ValueError):
        collect([str(tmp_path / "nowhere")])


def test_fold_summary_math():
    runs = [{"dir": "r0", "config": {"params": {"fold": 0}}, "history": [{"test/rmse": 1.0, "train/l": 5.0}]},
            {"dir": "r1", "config": {"params": {"fold": 1}}, "history": [{"test/rmse": 3.0, "train/l": 5.0}]}]
    result = fold_summary(runs)
    assert result["summary"] == {"test/rmse": {"mean": 2.0, "std": pytest.approx(2 ** 0.5)}}
    assert [fold["turns"] for fold in result["folds"]] == [1, 1]
    assert sweep_table(runs)["varying"] == ["fold"]


def test_set_rule_from_the_command_line(workdir, capsys):
    config = minimal()
    config["params"] = {"lr": 0.1}
    config["model"]["optimizer"]["params"]["lr"] = "$lr$"
    path = write_config(workdir / "cfg.yaml", config)
    assert main(["check", path, "-p", "lr=0.5", "--set", "device=cpu"]) == 0
    capsys.readouterr()
    assert main(["check", path, "--set", "training.epochs=abc"]) == 1
    assert "[invalid_value]" in capsys.readouterr().out
    with pytest.raises(SystemExit) as failure:
        main(["check", path, "--set", "lr=0.5"])
    assert failure.value.code == 2
    err = capsys.readouterr().err
    assert "-p lr=0.5" in err and "dotted path" in err
    with pytest.raises(SystemExit):
        main(["check", path, "--set", "nonsense"])


def test_check_recipe_prints_the_driver_document(workdir, config_01, capsys):
    from ruamel.yaml import YAML

    assert main(["check", config_01, "--recipe"]) == 0
    out = capsys.readouterr().out
    document = YAML(typ="safe").load(out.split("---\n", 1)[1])
    assert list(document) == ["losses", "metrics", "triggers", "plots", "progress", "blocks", "flow"]
