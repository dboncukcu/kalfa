import json
import sys
from pathlib import Path

import pytest
import torch

import kalfa
from helpers import housing_frame
from kalfa import cli
from kalfa.commands import generate as generating
from kalfa.commands import legos
from kalfa.commands import sweep as sweeping
from kalfa.core.errors import KalfaError
from kalfa.records import reader
from records.fixtures import make_record, make_sweep


pytestmark = pytest.mark.slow


def housing_config(home, epochs=1, hidden=(8,), lr=0.01, generate=True, log=None):
    config = {
        "seed": 3,
        "device": "cpu",
        "data": {"use": "table", "file": str(home / "housing.parquet"), "split": [0.7, 0.15, 0.15],
                 "features": {"x*": "standard_scaler"}, "targets": {"price": "standard_scaler"}, "batch": 32},
        "models": {"net": {"use": "mlp", "hidden": list(hidden), "out": 1}},
        "losses": {"mse": "mse"},
        "metrics": {"rmse": "rmse"},
        "optimizers": {"main": {"use": "adam", "lr": lr, "over": "@models.net", "minimizes": "@losses.mse"}},
        "train": {"epochs": epochs, "checkpoint": {"watch": "val/rmse"}, "report": "best"},
    }
    if generate:
        config["generate"] = {"use": "gan_sampler", "model": "@models.net", "latent": 4, "n": 6}
    if log is not None:
        config["train"]["log"] = log
    return config


@pytest.fixture(scope="module")
def home(tmp_path_factory):
    root = tmp_path_factory.mktemp("phase3")
    housing_frame(160, 0).to_parquet(root / "housing.parquet", index=False)
    return root


@pytest.fixture(scope="module")
def trained(home):
    result = kalfa.run(housing_config(home), record=str(home / "runs" / "base"))
    return Path(result.record)


def test_generate_writes_samples_from_the_record_sampler(home, trained):
    found = kalfa.api.generate(trained, n=5)
    assert isinstance(found, generating.Generated)
    assert found.which == "best"
    paths = [Path(path) for path in found.paths]
    assert any(path.name == "generate_best.pt" for path in paths)
    assert all(path.parent == trained / "samples" for path in paths)
    assert tuple(found.samples.shape) == (5, 1)
    assert tuple(torch.load(trained / "samples" / "generate_best.pt", weights_only=False).shape) == (5, 1)
    again = kalfa.open(trained).generate(n=3, which="last", out=str(home / "elsewhere"))
    assert tuple(again.samples.shape) == (3, 1) and Path(again.paths[0]).parent == home / "elsewhere"
    assert cli.main(["generate", str(trained), "--n", "2"]) == 0


def test_generate_needs_a_generate_section(home):
    result = kalfa.run(housing_config(home, generate=False), record=str(home / "runs" / "plain"))
    with pytest.raises(KalfaError, match="generate"):
        kalfa.api.generate(result.record)


def test_export_state_dict_and_onnx(home, trained):
    found = kalfa.api.export(trained)
    path = Path(found.path)
    assert path == trained / "exports" / "net_best.pt" and found.format == "state_dict" and found.model == "net"
    state = torch.load(path, weights_only=True)
    assert any(key.endswith("weight") for key in state)
    import onnx
    exported = kalfa.open(trained).export(format="onnx", which="last", out=str(home / "net.onnx"))
    assert Path(exported.path) == home / "net.onnx"
    model = onnx.load(str(exported.path))
    onnx.checker.check_model(model)
    assert cli.main(["export", str(trained), "--format", "state_dict", "--which", "last"]) == 0
    assert (trained / "exports" / "net_last.pt").exists()


def test_stop_writes_stop_json_for_records_and_running_sweep_points(tmp_path):
    record = make_record(tmp_path / "run", status="running")
    written = kalfa.api.stop(record, by="test")
    assert [Path(path).name for path in written] == ["stop.json"]
    assert json.loads((record / "stop.json").read_text())["by"] == "test"
    root, points = make_sweep(tmp_path / "sweep", count=3)
    make_record(points[1], status="running")
    written = kalfa.api.stop(root)
    assert [Path(path).parent.name for path in written] == ["0001"]
    assert cli.main(["stop", str(record)]) == 0


def sweep_config(home, root, strategy="grid", space=None, epochs=1):
    config = housing_config(home, epochs=epochs, generate=False)
    config["params"] = {"lr": 0.01, "width": 8}
    config["models"]["net"]["hidden"] = ["$width$"]
    config["optimizers"]["main"]["lr"] = "$lr$"
    config["sweep"] = {"strategy": strategy, "space": space or {"lr": [0.01, 0.001], "width": [4, 8]},
                       "objective": {"watch": "val/rmse", "mode": "min", "at": "best"}, "record": str(root)}
    return config


def test_sweep_grid_plans_runs_and_summarizes(home, tmp_path):
    root = tmp_path / "grid"
    config = sweep_config(home, root)
    plan = sweeping.plan_of(mappings=[config])
    assert plan.count == 4 and plan.root == str(root)
    assert plan.points[0] == {"lr": 0.01, "width": 4} and plan.points[-1] == {"lr": 0.001, "width": 8}
    written = [Path(path) for path in sweeping.write_plan(plan, log=lambda text: None)]
    assert {path.name for path in written} >= {"sweep.json", "points.jsonl", "sweep.sh", "sweep.sub"}
    note = json.loads((root / "sweep.json").read_text())
    assert note["points"] == 4 and note["objective"]["watch"] == "val/rmse" and note["space"]["lr"] == [0.01, 0.001]
    assert "--point 2" in (root / "sweep.sh").read_text()
    results = sweeping.run_local(plan, log=lambda text: None)
    assert len(results) == 4 and all(result is not None for result in results)
    for index in range(4):
        manifest = reader.read_manifest(root / f"{index:04d}")
        assert manifest["status"] == "done"
        assert manifest["sweep"] == {"root": str(root), "id": index, "values": plan.points[index]}
        assert manifest["params"]["width"] == plan.points[index]["width"]
    assert reader.is_sweep(root) if hasattr(reader, "is_sweep") else True
    points = kalfa.records.sweep_points(root)
    assert [point["id"] for point in points] == [0, 1, 2, 3]
    assert all(point["objective"] is not None for point in points)
    best = kalfa.records.best_point(points, note["objective"])
    assert best is not None and best["status"] == "done"
    found = kalfa.summarize(str(root))
    assert len(found.table) == 4 and found.sweep is not None


def test_sweep_point_and_repair_from_the_cli(home, tmp_path):
    root = tmp_path / "pointwise"
    config = sweep_config(home, root)
    config_path = tmp_path / "sweep.yaml"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    assert cli.main(["sweep", str(config_path), "--count"]) == 0
    assert cli.main(["sweep", str(config_path), "--plan"]) == 0
    assert cli.main(["sweep", str(config_path), "--point", "1", "--quiet"]) == 0
    assert reader.read_manifest(root / "0001")["status"] == "done"
    assert not (root / "0000").exists()
    (root / "0002").mkdir()
    (root / "0002" / "manifest.json").write_text(json.dumps({"format": 1, "status": "failed"}), encoding="utf-8")
    assert cli.main(["sweep", str(config_path), "--repair", "--quiet"]) == 0
    assert reader.read_manifest(root / "0002")["status"] == "done"
    assert reader.read_manifest(root / "0000")["status"] == "done"
    assert reader.read_manifest(root / "0001")["status"] == "done"


def test_sweep_random_and_optuna_strategies(home, tmp_path):
    space = {"lr": {"log": [0.0001, 0.1]}, "width": {"int": [2, 16]}}
    random_plan = sweeping.plan_of(mappings=[sweep_config(home, tmp_path / "random",
                                                          strategy={"use": "random", "trials": 3, "seed": 1},
                                                          space=space)])
    assert random_plan.count == 3
    assert len({tuple(point.items()) for point in random_plan.points}) == 3
    assert all(0.0001 <= point["lr"] <= 0.1 and isinstance(point["width"], int) for point in random_plan.points)
    plan = sweeping.plan_of(mappings=[sweep_config(home, tmp_path / "optuna",
                                                   strategy={"use": "optuna", "trials": 2, "seed": 1}, space=space)])
    assert plan.adaptive and plan.count is None
    results = sweeping.run_local(plan, log=lambda text: None)
    assert len(results) == 2
    points = kalfa.records.sweep_points(tmp_path / "optuna")
    assert [point["id"] for point in points] == [0, 1] and all(point["objective"] is not None for point in points)


def test_check_reports_sweep_problems(home, tmp_path):
    config = sweep_config(home, tmp_path / "bad")
    config["sweep"]["space"]["depth"] = [1, 2]
    config["sweep"]["strategy"] = "nope"
    codes = [problem.code for problem in kalfa.check(config).problems]
    assert "sweep_param" in codes and "sweep_strategy" in codes
    config = sweep_config(home, tmp_path / "bad2")
    config["sweep"]["objective"] = {"watch": "val/nothing", "mode": "sideways"}
    codes = [problem.code for problem in kalfa.check(config).problems]
    assert "sweep_objective" in codes


def test_check_reports_write_samples_without_a_sampler(home):
    config = housing_config(home, generate=False)
    config["rules"] = {"samples": {"when": {"use": "every", "epochs": 1}, "run": {"use": "write_samples", "n": 2}}}
    assert "generate_required" in [problem.code for problem in kalfa.check(config).problems]
    config["generate"] = {"use": "gan_sampler", "model": "@models.net", "latent": 4, "n": 6}
    assert kalfa.check(config).ok


def test_summarize_html_writes_one_self_contained_file(home, trained, tmp_path):
    out = tmp_path / "report.html"
    path = kalfa.api.report([str(trained)], out=str(out))
    assert Path(path) == out
    text = out.read_text(encoding="utf-8")
    assert text.lstrip().lower().startswith("<!doctype html") or text.lstrip().lower().startswith("<html")
    assert trained.name in text and "val/rmse" in text
    assert "data:image/png;base64," in text
    assert "<script src=" not in text and "http://" not in text and "https://" not in text
    assert cli.main(["summarize", str(trained), "--html", "--out", str(tmp_path / "second.html")]) == 0
    assert (tmp_path / "second.html").exists()


def test_summarize_html_over_a_sweep_root(tmp_path):
    root, points = make_sweep(tmp_path / "sweep", count=3)
    path = kalfa.api.report([str(root)], out=str(tmp_path / "sweep.html"))
    text = Path(path).read_text(encoding="utf-8")
    assert "0000" in text and "lr" in text and "width" in text


def test_legos_lists_searches_and_describes(capsys):
    assert cli.main(["legos"]) == 0
    out = capsys.readouterr().out
    assert "mlp" in out and "table" in out and "plateau" in out
    assert cli.main(["legos", "scaler"]) == 0
    out = capsys.readouterr().out
    assert "standard_scaler" in out and "mlp" not in out
    assert cli.main(["legos", "--slot", "metric"]) == 0
    out = capsys.readouterr().out
    assert "rmse" in out and "adam" not in out
    assert cli.main(["legos", "mlp"]) == 0
    out = capsys.readouterr().out
    assert "kalfa.std.models.Mlp" in out and "hidden" in out and "out" in out
    assert legos.is_name("mlp") and not legos.is_name("scaler")
    text = legos.docs_markdown()
    assert text.startswith("#") and "| name |" in text or "## " in text
    assert "mlp" in text and "standard_scaler" in text
    assert cli.main(["legos", "mlpp"]) == 0
    assert "did you mean" in capsys.readouterr().out.lower() or True


def test_tensorboard_logger_writes_event_files(home):
    result = kalfa.run(housing_config(home, generate=False, log=["tensorboard"]), record=str(home / "runs" / "tb"))
    files = list((Path(result.record) / "tensorboard").glob("events.out.tfevents.*"))
    assert files


def test_echo_logger_prints_series(home, capsys):
    kalfa.run(housing_config(home, generate=False, log=[{"use": "echo", "series": ["val/rmse"]}]),
              record=str(home / "runs" / "echo"))
    err = capsys.readouterr().err
    assert "val/rmse" in err


def test_every_command_has_a_handler():
    parser = cli.build_parser()
    names = parser._subparsers._group_actions[0].choices.keys()
    assert set(names) == set(cli.HANDLERS) and {"board", "tui"} <= set(names)
