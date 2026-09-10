import pytest
from ruamel.yaml import YAML

from helpers import example, minimal, write_config
from kalfa.cli import main


def test_check_reports_problems_and_exit_codes(workdir, capsys):
    path = write_config(workdir / "cfg.yaml", minimal())
    assert main(["check", path]) == 0
    out = capsys.readouterr().out
    assert out.strip() == "no problems found"
    config = minimal()
    config["training"]["checkpoint"] = "last"
    path = write_config(workdir / "bad.yaml", config)
    assert main(["check", path]) == 1
    out = capsys.readouterr().out
    assert "[report_mismatch]" in out and "bad.yaml:" in out


def test_check_layers_prints_the_tree(workdir, capsys):
    base = write_config(workdir / "base.yaml", minimal())
    top = workdir / "top.yaml"
    top.write_text("include: [base.yaml]\ntraining:\n  epochs: 2\n")
    assert main(["check", str(top), "--layers", "-p", "lr=0.5", "--set", "training.epochs=3"]) == 0
    out = capsys.readouterr().out
    assert "layers, bottom to top:" in out and "base.yaml" in out and "--set" in out
    assert "overrides training.epochs" in out


def test_check_load_reports_the_real_sizes(workdir, capsys):
    path = write_config(workdir / "cfg.yaml", minimal())
    assert main(["check", path, "--load"]) == 0
    printed = capsys.readouterr().out
    assert "loaded the data block:" in printed and "housing.parquet 2 000 rows" in printed
    assert "fitted std_scaler, target_std on train ─→ table feed ─→ 3 loaders, batch 128" in printed
    assert "sets after filters: train 1 400" in printed


def test_set_rule_from_the_command_line(workdir, capsys):
    path = write_config(workdir / "cfg.yaml", minimal())
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


def test_check_recipe_prints_the_driver_document(workdir, capsys):
    assert main(["check", example("01_mlp_regression"), "--recipe"]) == 0
    out = capsys.readouterr().out
    document = YAML(typ="safe").load(out.split("---\n", 1)[1])
    assert list(document) == ["losses", "metrics", "triggers", "plots", "calibrate", "checkpoint", "blocks", "flow"]


def test_check_dump_prints_the_flow(workdir, capsys):
    assert main(["check", example("01_mlp_regression"), "--dump"]) == 0
    out = capsys.readouterr().out
    document = YAML(typ="safe").load(out.split("---\n", 1)[1])
    assert list(document) == ["components", "blocks", "flow"]
    assert "# implicit: device" in out
