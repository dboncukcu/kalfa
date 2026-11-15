import json
from pathlib import Path

import numpy
import pandas
import pytest

import kalfa
from helpers import CLASSES, blobs_frame, housing_frame, write_parquet
from kalfa.core import run as running
from kalfa.core.state import read_checkpoint
from kalfa.records.reader import read_lines


RECORD_FILES = ("manifest.json", "resolved.yaml", "plan.json", "history.jsonl", "updates.jsonl", "events.jsonl",
                "predictions.parquet", "model_summary.json", "fitted/fields.json", "fitted/preprocessors.joblib",
                "checkpoints/last.pt", "checkpoints/best.pt", "checkpoints/final.pt", "plots/loss_curve.png",
                "plots/pred_vs_true.png", "plugins/integration_legos.py")

LEGOS = """import torch


def log_cosh(pred, target):
    return torch.log(torch.cosh(pred - target)).mean()
"""


def regression_config(record, epochs=2):
    return {
        "params": {"epochs": epochs, "lr": 1.0e-3, "seed": 7},
        "seed": "$seed$",
        "device": "cpu",
        "data": {"use": "table", "file": "housing.parquet", "split": [0.7, 0.15, 0.15],
                 "features": {"x*": "standard_scaler"}, "targets": {"price": "standard_scaler"}, "batch": 64},
        "models": {"net": {"use": "mlp", "hidden": [32, 32], "out": 1}},
        "losses": {"mse": "mse", "huber": {"use": "torch.nn.HuberLoss", "delta": 1.0}, "mae": "torch.nn.L1Loss",
                   "log_cosh": "integration_legos.log_cosh"},
        "metrics": {"rmse": "rmse", "mae": "mae"},
        "optimizers": {"main": {"use": "adam", "lr": "$lr$", "over": "@models.net", "minimizes": "@losses.mse"}},
        "train": {"epochs": "$epochs$", "checkpoint": {"watch": "val/rmse"}, "report": "best",
                  "track": ["losses.huber.delta", "optimizers.main.lr"]},
        "rules": {
            "to_huber": {"when": {"use": "after_epoch", "at": 1},
                         "set": {"optimizers.main.minimizes": "@losses.huber", "losses.huber.delta": 2.0,
                                 "optimizers.main.lr": {"times": 0.5}}},
            "to_mae": {"after": "to_huber", "when": {"use": "below", "watch": "train/huber", "value": 0.01},
                       "set": {"optimizers.main.minimizes": "@losses.mae"}},
            "stop": {"when": {"use": "plateau", "watch": "val/rmse", "patience": 6}, "stop": True},
        },
        "plots": {"loss_curve": "loss_curve", "pred_vs_true": "pred_vs_true"},
        "record": str(record),
    }


def classification_config(file, record):
    return {
        "seed": 3,
        "device": "cpu",
        "data": {"use": "table", "file": str(file), "split": [0.6, 0.2, 0.2], "features": {"f*": "standard_scaler"},
                 "targets": {"kind": "label"}, "spectators": ["id"], "batch": 32},
        "models": {"net": {"use": "mlp", "hidden": [16], "out": "@data.n_classes", "in_features": 2,
                           "freeze": ["layers.0.*"], "ema": 0.9}},
        "losses": {"ce": {"use": "cross_entropy", "weight": "@data.class_weights"}},
        "metrics": {"acc": "accuracy", "f1": {"use": "f1", "num_classes": "@data.n_classes"}},
        "optimizers": {"main": {"use": "adam", "lr": 0.01, "over": "@models.net", "minimizes": "@losses.ce"}},
        "train": {"epochs": 2, "checkpoint": {"watch": "val/acc", "mode": "max"}},
        "rules": {"thaw": {"when": {"use": "after_epoch", "at": 1}, "set": {"models.net.freeze": False}}},
        "plots": {"confusion_matrix": "confusion_matrix"},
        "record": str(record),
    }


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    home = tmp_path_factory.mktemp("integration")
    write_parquet(housing_frame(), home / "housing.parquet")
    (home / "integration_legos.py").write_text(LEGOS, encoding="utf-8")
    config = home / "config.yaml"
    config.write_text(json.dumps(regression_config(home / "runs" / "housing")), encoding="utf-8")
    return home


@pytest.fixture(scope="module")
def trained(workspace):
    return kalfa.run(workspace / "config.yaml")


def history_of(record):
    return read_lines(Path(record) / "history.jsonl")


def test_check_of_the_dict_config_reports_series_and_mutable_paths(workspace, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(workspace))
    config = regression_config(tmp_path / "record")
    config["data"]["file"] = str(workspace / "housing.parquet")
    checked = kalfa.check(config)
    assert checked.ok, checked.text()
    assert [(problem.code, problem.path) for problem in checked.warnings] == [("series_clash", "metrics.mae")]
    plan = checked.plan
    assert plan["nodes"]["losses.log_cosh"] == {
        "slot": "loss", "use": "integration_legos.log_cosh", "params": {}, "keys": {}, "fixed": ["pred", "target"],
        "signature": "checked"}
    assert {"train/mse", "val/rmse", "test/mae", "lr/main", "minimizes/main", "track/losses.huber.delta",
            "track/optimizers.main.lr"} <= set(plan["series"])
    assert {"losses.huber.delta", "optimizers.main.lr", "optimizers.main.minimizes", "models.net.freeze",
            "train.epochs", "losses.mse.on"} <= set(plan["mutable"])
    assert plan["rules"][0]["patch"][0] == {"path": "optimizers.main.minimizes", "value": {"ref": "losses.huber"}}


def test_two_turns_write_the_record_with_defaults_patches_and_tracks(workspace, trained):
    record = Path(trained.record)
    assert record == workspace / "runs" / "housing"
    for name in RECORD_FILES:
        assert (record / name).exists(), name
    plan = json.loads((record / "plan.json").read_text(encoding="utf-8"))
    assert plan["nodes"]["losses.mse"]["keys"] == {"pred": "net", "target": "price"}
    assert plan["nodes"]["losses.log_cosh"]["keys"] == {"pred": "net", "target": "price"}
    assert plan["nodes"]["metrics.rmse"]["keys"] == {"pred": "net", "target": "price", "rescale": True}
    assert plan["nodes"]["optimizers.main"]["params"]["lr"] == pytest.approx(0.0005)
    assert plan["nodes"]["losses.huber"]["params"]["delta"] == 2.0
    assert plan["nodes"]["optimizers.main"]["keys"]["minimizes"] == {"ref": "losses.huber"}
    history = history_of(record)
    assert [line["turn"] for line in history] == [1, 2]
    assert history[0]["rules"] == ["to_huber"] and history[1]["rules"] == []
    assert [line["minimizes/main"] for line in history] == ["huber", "huber"]
    assert [line["track/losses.huber.delta"] for line in history] == [1.0, 2.0]
    assert [line["track/optimizers.main.lr"] for line in history] == [pytest.approx(0.001), pytest.approx(0.0005)]
    assert history[0]["lr/main"] == pytest.approx(0.0005)
    assert {"train/mse", "train/mae", "train/log_cosh", "val/rmse", "val/mae", "test/rmse"} <= set(history[0])
    assert "train/huber" in history[1] and "train/mse" in history[1]
    events = read_lines(record / "events.jsonl")
    patches = [event for event in events if event["kind"] == "patch"]
    assert [(event["path"], event["value"], event["turn"]) for event in patches] == [
        ("optimizers.main.minimizes", {"ref": "losses.huber"}, 1), ("losses.huber.delta", 2.0, 1),
        ("optimizers.main.lr", pytest.approx(0.0005), 1)]
    assert patches[2]["asked"] == {"times": 0.5}
    ctx = trained.context
    assert ctx.optimizers["main"].minimizes is ctx.losses["huber"]
    assert ctx.losses["huber"].function.delta == 2.0 and ctx.optimizers["main"].rate() == pytest.approx(0.0005)
    assert ctx.losses["mse"].rescale is False and ctx.metrics["rmse"].rescale is True
    manifest = json.loads((record / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "done" and manifest["turns_planned"] == 2 and manifest["params"]["epochs"] == 2
    resolved = (record / "resolved.yaml").read_text(encoding="utf-8")
    assert "epochs: 2" in resolved and "$epochs$" not in resolved
    predictions = pandas.read_parquet(record / "predictions.parquet")
    assert {"row", "price", "pred_net", "raw_net"} <= set(predictions.columns)
    assert len(predictions) == 30 and numpy.isfinite(predictions["pred_net"].to_numpy()).all()
    assert predictions["pred_net"].abs().mean() > 10 * predictions["raw_net"].abs().mean()
    checkpoint = read_checkpoint(record / "checkpoints" / "final.pt")
    assert checkpoint["turn"] == 2 and {"models.net", "optimizers.main", "rules.to_huber"} <= set(checkpoint["nodes"])
    assert checkpoint["nodes"]["rules.to_huber"]["fired_turn"] == 1


def test_resume_replays_the_patches_and_continues_to_turn_three(trained, tmp_path):
    target = tmp_path / "resumed"
    result = running.resume(trained.record, sets=[("train.epochs", 3)], target=target)
    assert Path(result.record) == target and result.context.turn == 3
    history = history_of(target)
    assert [line["turn"] for line in history] == [1, 2, 3]
    assert history[2]["minimizes/main"] == "huber" and history[2]["rules"] == []
    assert history[2]["track/losses.huber.delta"] == 2.0
    assert history[2]["lr/main"] == pytest.approx(0.0005)
    ctx = result.context
    assert ctx.optimizers["main"].minimizes is ctx.losses["huber"] and ctx.losses["huber"].function.delta == 2.0
    assert ctx.rules_by_name["to_huber"].fired_turn == 1 and ctx.rules_by_name["to_huber"].done
    events = read_lines(target / "events.jsonl")
    kinds = [event["kind"] for event in events]
    assert "resume" in kinds and "patch" not in kinds and kinds[-1] == "run_end"
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "done" and manifest["resumed_from"] == str(trained.record)
    assert manifest["params"] == {"epochs": 2, "lr": 1.0e-3, "seed": 7}
    assert read_checkpoint(target / "checkpoints" / "final.pt")["turn"] == 3
    assert len(pandas.read_parquet(target / "predictions.parquet")) == 30


def test_open_record_predicts_on_new_rows_and_on_the_test_set(trained):
    record = Path(trained.record)
    opened = kalfa.open(record)
    rows = housing_frame(20, seed=3)
    table = opened.predict(data=rows)
    assert list(table["row"]) == list(range(20))
    assert {"price", "pred_net", "raw_net"} <= set(table.columns)
    assert numpy.allclose(table["price"].to_numpy(), rows["price"].to_numpy())
    assert numpy.isfinite(table["pred_net"].to_numpy()).all()
    assert (table["pred_net"].between(100, 300)).all() and (table["raw_net"].abs() < 10).all()
    assert (record / "predictions_frame.parquet").exists()
    again = opened.predict()
    assert len(again) == 30 and {"row", "price", "pred_net"} <= set(again.columns)
    assert opened.best["series"] == "val/rmse" and opened.best["turn"] in (1, 2)


def test_classification_decodes_labels_freezes_then_thaws_and_keeps_an_ema(tmp_path):
    file = write_parquet(blobs_frame(), tmp_path / "blobs.parquet")
    record = tmp_path / "record"
    config = classification_config(file, record)
    checked = kalfa.check(config)
    assert checked.ok, checked.text()
    result = kalfa.run(config)
    ctx = result.context
    history = history_of(record)
    assert [line["turn"] for line in history] == [1, 2]
    assert {"train/ce", "val/acc", "val/f1", "test/acc"} <= set(history[0])
    assert history[0]["rules"] == ["thaw"]
    assert ctx.models["net"].freeze is False
    assert all(parameter.requires_grad for parameter in ctx.models["net"].module.parameters())
    assert ctx.models["net"].module.layers[-1].out_features == 3
    assert ctx.losses["ce"].function.weight.shape == (3,)
    assert ctx.metrics["acc"].rescale is False and ctx.metrics["f1"].metric.num_classes == 3
    events = read_lines(record / "events.jsonl")
    assert [(event["path"], event["value"]) for event in events if event["kind"] == "patch"] == [
        ("models.net.freeze", False)]
    checkpoint = read_checkpoint(record / "checkpoints" / "last.pt")
    assert {"models.net", "models.net.ema", "optimizers.main", "rules.thaw"} <= set(checkpoint["nodes"])
    assert checkpoint["nodes"]["models.net.ema"]["_extra_state"]["updates"] == ctx.updates == 10
    fields = json.loads((record / "fitted" / "fields.json").read_text(encoding="utf-8"))
    assert set(fields["classes"]) == set(CLASSES) and fields["targets"] == ["kind"]
    predictions = pandas.read_parquet(record / "predictions.parquet")
    assert {"row", "kind", "pred_net", "raw_net", "id"} <= set(predictions.columns)
    assert set(predictions["kind"]) <= set(CLASSES) and set(predictions["pred_net"]) <= set(CLASSES)
    assert set(predictions["raw_net"].tolist()) <= {0, 1, 2}
    assert (record / "plots" / "confusion_matrix.png").exists()
    fresh = blobs_frame(12, seed=9)
    table = kalfa.open(record).predict(data=fresh)
    assert len(table) == 12 and set(table["pred_net"]) <= set(CLASSES)
    assert list(table["kind"]) == list(fresh["kind"]) and list(table["id"]) == list(fresh["id"])
