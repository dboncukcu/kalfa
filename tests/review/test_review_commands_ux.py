import json
import os
import sys
from pathlib import Path

import pytest

import kalfa
from helpers import housing_frame
from kalfa import cli
from kalfa.core.errors import KalfaError
from records.fixtures import make_sweep


pytestmark = pytest.mark.slow


def plain_config(home):
    return {
        "seed": 3,
        "device": "cpu",
        "data": {"use": "table", "file": str(home / "housing.parquet"), "split": [0.7, 0.15, 0.15],
                 "features": {"x*": "standard_scaler"}, "targets": {"price": "standard_scaler"}, "batch": 32},
        "models": {"net": {"use": "mlp", "hidden": [8], "out": 1}},
        "losses": {"mse": "mse"},
        "metrics": {"rmse": "rmse"},
        "optimizers": {"main": {"use": "adam", "lr": 0.01, "over": "@models.net", "minimizes": "@losses.mse"}},
        "train": {"epochs": 1, "checkpoint": {"watch": "val/rmse"}, "report": "best"},
    }


def sweep_config(home, root):
    config = plain_config(home)
    config["params"] = {"lr": 0.01, "width": 8}
    config["models"]["net"]["hidden"] = ["$width$"]
    config["optimizers"]["main"]["lr"] = "$lr$"
    config["sweep"] = {"strategy": "grid", "space": {"lr": [0.01, 0.001], "width": [4]},
                       "objective": {"watch": "val/rmse", "mode": "min", "at": "best"}, "record": str(root)}
    return config


@pytest.fixture(scope="module")
def home(tmp_path_factory):
    root = tmp_path_factory.mktemp("review_commands")
    housing_frame(160, 0).to_parquet(root / "housing.parquet", index=False)
    return root


@pytest.fixture(scope="module")
def plain(home):
    return Path(kalfa.run(plain_config(home), record=str(home / "runs" / "plain")).record)


def test_generate_without_a_section_names_the_record_and_the_section(plain, capsys):
    with pytest.raises(KalfaError, match="has no generate section"):
        kalfa.api.generate(plain)
    assert cli.main(["generate", str(plain)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ") and str(plain) in err and "generate section" in err
    assert not (plain / "samples").exists()


def test_export_onnx_without_the_package_names_the_dependency(plain, capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "onnx", None)
    assert cli.main(["export", str(plain), "--format", "onnx"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "onnx package" in err and "uv sync" in err
    assert not (plain / "exports").exists()


def test_export_and_generate_agree_on_the_loaded_checkpoint(plain):
    found = kalfa.api.export(plain)
    assert found.which == "best" and Path(found.path).name == "net_best.pt"
    with pytest.raises(KalfaError, match="format must be one of"):
        kalfa.api.export(plain, format="pickle")


def test_sweep_count_without_a_sweep_section_says_so(home, tmp_path, capsys):
    path = tmp_path / "plain.yaml"
    path.write_text(json.dumps(plain_config(home)), encoding="utf-8")
    assert cli.main(["sweep", str(path), "--count"]) == 2
    err = capsys.readouterr().err
    assert "no sweep section" in err and "strategy" in err and "record" in err


def test_sweep_check_names_a_strategy_typo_with_a_suggestion(home, tmp_path, capsys):
    config = sweep_config(home, tmp_path / "grid")
    config["sweep"]["strategy"] = "gird"
    path = tmp_path / "typo.yaml"
    path.write_text(json.dumps(config), encoding="utf-8")
    assert cli.main(["sweep", str(path), "--count"]) == 1
    err = capsys.readouterr().err
    assert "sweep_strategy" in err and "'gird' is not a sweep strategy" in err and "did you mean grid" in err


def test_stop_refuses_a_path_that_is_neither_record_nor_root(tmp_path, capsys):
    assert cli.main(["stop", str(tmp_path)]) == 2
    assert "neither a record directory nor a sweep root" in capsys.readouterr().err
    root, points = make_sweep(tmp_path / "sweep", count=2)
    (root / "0002").mkdir()
    assert kalfa.api.stop(root) == []
    assert not any((point / "stop.json").exists() for point in points)


def test_legos_misses_suggest_and_list_the_slots(capsys):
    assert cli.main(["legos", "nosuch"]) == 0
    assert capsys.readouterr().out.splitlines()[0] == "no name matches nosuch"
    assert cli.main(["legos", "mlpp"]) == 0
    assert "did you mean mlp" in capsys.readouterr().out
    assert cli.main(["legos", "--slot", "metrics"]) == 0
    out = capsys.readouterr().out
    assert "no name of slot metrics" in out and "metric" in out and "strategy" in out
    assert cli.main(["legos", "grid"]) == 0
    out = capsys.readouterr().out
    assert "kalfa.commands.sweep.Grid" in out and "steps = None" in out


def test_sweep_plan_files_name_every_point(home, tmp_path, capsys):
    root = tmp_path / "grid"
    path = tmp_path / "sweep.yaml"
    path.write_text(json.dumps(sweep_config(home, root)), encoding="utf-8")
    assert cli.main(["sweep", str(path), "--show", "5"]) == 0
    assert capsys.readouterr().out.strip().splitlines() == ["0000  lr=0.01  width=4", "0001  lr=0.001  width=4"]
    assert cli.main(["sweep", str(path), "--plan"]) == 0
    printed = capsys.readouterr().out
    assert "2 points" in printed and str(root / "sweep.sh") in printed
    commands = [line for line in (root / "sweep.sh").read_text(encoding="utf-8").splitlines()
                if line.startswith("kalfa sweep ")]
    assert len(commands) == 2
    for index, line in enumerate(commands):
        assert f" --point {index} " in line and os.path.abspath(path) in line and f" --record {root} " in line
    submit = (root / "sweep.sub").read_text(encoding="utf-8")
    assert "queue 2" in submit and "--point $(Process)" in submit and f"{root / 'logs'}" in submit
    assert len((root / "points.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    assert (root / "logs").is_dir() and not (root / "0000").exists()
    assert cli.main(["sweep", str(path), "--point", "5", "--quiet"]) == 2
    assert "point 5 is not in the plan" in capsys.readouterr().err


def test_report_over_a_root_has_every_section_and_names_a_missing_point(tmp_path, capsys):
    root, points = make_sweep(tmp_path / "sweep", count=3)
    path = Path(kalfa.api.report([str(root)], out=str(tmp_path / "out")))
    assert path == tmp_path / "out" / "report.html"
    text = path.read_text(encoding="utf-8")
    for key in ("summary", "curves", "sweep", "records"):
        assert f'<h2 id="{key}">' in text and f'href="#{key}"' in text
    assert "data:image/png;base64," in text and "<script" not in text
    assert text.count("<h3>") == 3 and "best point 0000" in text
    with pytest.raises(KalfaError, match="no record or point 7 in the report; it holds 0000, 0001, 0002"):
        kalfa.api.report([str(root)], out=str(tmp_path / "x.html"), points="7")
    assert cli.main(["summarize", str(root), "--html", "--out", str(tmp_path / "x.html"), "--points", "0,2"]) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "x.html")
    text = (tmp_path / "x.html").read_text(encoding="utf-8")
    assert text.count("<h3>") == 2 and "<h3>0001</h3>" not in text
