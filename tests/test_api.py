import io
import json
import os
import signal
import time
from pathlib import Path
from types import SimpleNamespace

import numpy
import pandas
import pytest

import kalfa
from helpers import housing_frame
from kalfa import api
from kalfa import summary as summarizing
from kalfa.core import run as running
from kalfa.core.check import Checked
from kalfa.core.errors import ProblemsError, RunError, error
from kalfa.records.reader import read_lines
from records.fixtures import make_record, make_sweep


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "01_mlp_regression"
RECORD_FILES = ("manifest.json", "resolved.yaml", "plan.json", "history.jsonl", "updates.jsonl", "events.jsonl",
                "predictions.parquet", "model_summary.json")


def housing_run(file, record, epochs=2):
    return kalfa.Run(
        params={"epochs": epochs, "lr": 1.0e-3, "seed": 7},
        seed="$seed$",
        device="cpu",
        data=kalfa.call("table", file=str(file), split=[0.7, 0.15, 0.15], features={"x*": "standard_scaler"},
                        targets={"price": "standard_scaler"}, batch=128),
        models={"net": kalfa.call("mlp", hidden=[128, 128], out=1)},
        losses={"mse": "mse", "huber": kalfa.call("torch.nn.HuberLoss", delta=1.0), "mae": "torch.nn.L1Loss",
                "log_cosh": "mylegos.log_cosh"},
        metrics={"rmse": "rmse", "mae": "mae"},
        optimizers={"main": kalfa.call("adam", lr="$lr$", over="@models.net", minimizes="@losses.mse")},
        train=kalfa.Train(epochs="$epochs$", checkpoint={"watch": "val/rmse"}, report="best"),
        rules={
            "to_huber": kalfa.Rule(when=kalfa.call("after_epoch", at=1),
                                   set={"optimizers.main.minimizes": "@losses.huber"}),
            "stop": kalfa.Rule(when=kalfa.call("plateau", watch="val/rmse", patience=6), stop=True),
        },
        plots={"loss_curve": "loss_curve", "pred_vs_true": "pred_vs_true"},
        record=str(record),
    )


@pytest.fixture(scope="module")
def housing(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "housing.parquet"
    housing_frame().to_parquet(path, index=False)
    return path


@pytest.fixture(scope="module")
def trained(housing, tmp_path_factory):
    patch = pytest.MonkeyPatch()
    patch.syspath_prepend(str(EXAMPLE))
    record = tmp_path_factory.mktemp("runs") / "housing"
    try:
        yield housing_run(housing, record).start()
    finally:
        patch.undo()


class FakeRecord:
    def __init__(self):
        self.directory = Path("fake")
        self.manifests = []
        self.events = []
        self.lines = []
        self.logged = []

    def manifest(self, **changes):
        self.manifests.append(changes)

    def log(self, text):
        self.logged.append(str(text))

    def beat(self):
        return True

    def event(self, kind, turn=None, **fields):
        self.events.append({"kind": kind, "turn": turn, **fields})

    def append(self, name, line):
        self.lines.append((name, line))


class FakeContext:
    def __init__(self):
        self.record = FakeRecord()
        self.history = []
        self.stopping = None
        self.turn = 0
        self.monitor = running.Monitor(quiet=True)

    def stop(self, reason=None):
        self.stopping = reason or "stop"


def test_call_builds_a_use_mapping():
    assert kalfa.call("adam", lr=0.1, over="@models.net") == {"use": "adam", "lr": 0.1, "over": "@models.net"}


def test_run_to_mapping_converts_train_and_rules():
    mapping = housing_run("housing.parquet", "runs/x").to_mapping()
    assert mapping["train"] == {"epochs": "$epochs$", "checkpoint": {"watch": "val/rmse"}, "report": "best"}
    assert mapping["rules"]["to_huber"] == {"when": {"use": "after_epoch", "at": 1},
                                            "set": {"optimizers.main.minimizes": "@losses.huber"}}
    assert mapping["rules"]["stop"] == {"when": {"use": "plateau", "watch": "val/rmse", "patience": 6},
                                        "stop": True}
    assert mapping["data"]["use"] == "table" and mapping["record"] == "runs/x"
    assert mapping["losses"]["huber"] == {"use": "torch.nn.HuberLoss", "delta": 1.0}
    assert kalfa.Rule(after="a", once=False, run={"use": "x"}).to_mapping() == {"after": "a", "once": False,
                                                                                 "run": {"use": "x"}}


def test_default_record_names_the_first_config():
    assert running.default_record(["a/b/housing.yaml", "x.yaml"]).startswith("runs/housing_")
    assert running.default_record([]).startswith("runs/run_")
    assert running.default_record(["<mapping 1>"]).startswith("runs/run_")


def test_monitor_prints_turn_seconds_and_four_series():
    stream = io.StringIO()
    monitor = running.Monitor(stream=stream)
    monitor.expect(5)
    monitor.turn({"turn": 2, "update": 10, "train/mse": 0.5, "val/rmse": 0.25, "test/rmse": 0.3, "lr/main": 0.001,
                  "minimizes/main": "mse", "track/x": 1.0, "seconds": 1.26, "rules": []})
    text = stream.getvalue()
    assert text.startswith("turn 2/5  1.3s  train/mse 0.5  val/rmse 0.25  test/rmse 0.3  lr/main 0.001")
    assert "track/x" not in text and "minimizes" not in text and "update" not in text
    quiet = running.Monitor(stream=stream, quiet=True)
    quiet.turn({"turn": 3})
    assert stream.getvalue() == text
    assert running.progress_text({"turn": 1}) == "turn 1"


def test_pairs_accept_strings_pairs_and_mappings():
    found = running.pairs(["train.epochs=3", ("seed", 1), {"device": "cpu"}])
    assert found == [("train.epochs", 3), ("seed", 1), ("device", "cpu")]
    assert running.pairs(["lr=1e-4"], "param") == [("lr", 1e-4)]
    assert running.pairs({"train.epochs": 3, "device": "cpu"}) == [("train.epochs", 3), ("device", "cpu")]
    assert running.pairs(None) == []
    with pytest.raises(kalfa.KalfaError):
        running.pairs(["nonsense"])
    with pytest.raises(kalfa.KalfaError):
        running.pairs([3])


def test_prepare_raises_problems_error_on_errors():
    checked = Checked(config=None, tree=None, plan=None,
                      problems=[error("unknown_key", "no such key", "models.net")])
    with pytest.raises(ProblemsError) as caught:
        running.prepare(checked, record=False)
    assert caught.value.problems[0].code == "unknown_key"
    assert "unknown_key" in str(caught.value)


def test_fill_defaults_names_pred_target_and_rescale():
    class Data:
        target_names = ("price",)

        def rescales(self, name):
            return name == "price"

    class Entry:
        def __init__(self, rescale=None):
            self.pred = None
            self.target = None
            self.rescale = rescale

    model = SimpleNamespace(outputs=["net"])
    ctx = SimpleNamespace(data=Data(), models={"net": model}, losses={"mse": Entry(False)},
                          metrics={"rmse": Entry()}, train=SimpleNamespace(predicts=None),
                          tree=SimpleNamespace(nodes={}), plan={"nodes": {"metrics.rmse": {"keys": {}}}},
                          predicts=lambda: model)
    running.fill_defaults(ctx)
    assert (ctx.losses["mse"].pred, ctx.losses["mse"].target, ctx.losses["mse"].rescale) == ("net", "price", False)
    assert (ctx.metrics["rmse"].pred, ctx.metrics["rmse"].target, ctx.metrics["rmse"].rescale) == ("net", "price",
                                                                                                      True)
    assert ctx.plan["nodes"]["metrics.rmse"]["keys"] == {"pred": "net", "target": "price", "rescale": True}
    ctx.data.target_names = ("price", "rooms")
    ctx.losses["mse"] = Entry(False)
    with pytest.raises(RunError, match="target_required"):
        running.fill_defaults(ctx)
    ctx.losses["mse"] = Entry(False)
    ctx.data.target_names = ("price",)
    ctx.predicts = lambda: None
    ctx.models = {}
    with pytest.raises(RunError, match="pred_required"):
        running.fill_defaults(ctx)


def test_train_into_marks_done_stopped_and_failed(monkeypatch):
    ctx = FakeContext()
    monkeypatch.setattr(running.loop, "train", lambda ctx: [{"turn": 1}])
    monkeypatch.setattr(running.loop, "after", lambda ctx: None)
    result = running.train_into(ctx)
    assert result.history == [{"turn": 1}] and result.context is ctx and result.record == "fake"
    assert ctx.record.manifests[-1]["status"] == "done" and ctx.record.manifests[-1]["finished"]
    ctx.stopping = "rule stop"
    running.train_into(ctx)
    assert ctx.record.manifests[-1]["status"] == "stopped"

    def failing(ctx):
        raise RuntimeError("boom")

    monkeypatch.setattr(running.loop, "train", failing)
    with pytest.raises(RuntimeError, match="boom"):
        running.train_into(ctx)
    assert ctx.record.manifests[-1]["status"] == "failed"
    assert ctx.record.manifests[-1]["failure"]["type"] == "RuntimeError"
    assert "boom" in ctx.record.manifests[-1]["failure"]["traceback"]
    assert ctx.record.events[-1]["kind"] == "error"
    logged = ctx.record.logged
    assert logged[0].startswith("run on ") and "done after turn 0" in logged
    assert "stopped: rule stop after turn 0" in logged and "failed in turn 0: RuntimeError: boom" in logged
    assert logged[-1].startswith("Traceback") and logged[-1].endswith("RuntimeError: boom")

    def interrupted(ctx):
        raise KeyboardInterrupt

    monkeypatch.setattr(running.loop, "train", interrupted)
    with pytest.raises(KeyboardInterrupt):
        running.train_into(ctx)
    assert ctx.record.manifests[-1]["status"] == "stopped"
    assert ctx.record.events[-1]["kind"] == "stop"


def test_first_interrupt_asks_to_stop_and_second_aborts():
    ctx = FakeContext()
    previous = signal.getsignal(signal.SIGINT)
    with running.interruptible(ctx):
        os.kill(os.getpid(), signal.SIGINT)
        time.sleep(0.05)
        assert ctx.stopping == "keyboard"
        with pytest.raises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGINT)
            time.sleep(0.05)
    assert signal.getsignal(signal.SIGINT) is previous


def test_open_wraps_a_directory_and_open_record_needs_resolved_yaml(tmp_path):
    record = kalfa.open(tmp_path)
    assert isinstance(record, kalfa.Record) and record.directory == str(tmp_path)
    with pytest.raises(kalfa.KalfaError, match="resolved.yaml"):
        running.open_record(tmp_path)


@pytest.mark.slow
def test_run_writes_the_record(trained):
    record = Path(trained.record)
    for name in RECORD_FILES:
        assert (record / name).exists(), name
    assert (record / "checkpoints" / "last.pt").exists()
    assert (record / "checkpoints" / "final.pt").exists()
    assert (record / "fitted").is_dir()
    manifest = json.loads((record / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "done"
    history = read_lines(record / "history.jsonl")
    assert [line["turn"] for line in history] == [1, 2]
    assert history[1]["minimizes/main"] == "huber"
    assert "val/rmse" in history[0] and "train/mse" in history[0]
    events = read_lines(record / "events.jsonl")
    patches = [event for event in events if event["kind"] == "patch"]
    assert patches and patches[0]["path"] == "optimizers.main.minimizes" and patches[0]["turn"] == 1
    assert {"run_start", "run_end"} <= {event["kind"] for event in events}
    predictions = pandas.read_parquet(record / "predictions.parquet")
    assert {"pred_net", "price"} <= set(predictions.columns)
    assert len(trained.history) == 2 and trained.context.turn == 2


@pytest.mark.slow
def test_resume_continues_in_a_new_directory(trained, tmp_path):
    target = tmp_path / "resumed"
    result = running.resume(trained.record, sets=[("train.epochs", 4)], target=target)
    assert Path(result.record) == target
    history = read_lines(target / "history.jsonl")
    assert [line["turn"] for line in history] == [1, 2, 3, 4]
    assert history[-1]["minimizes/main"] == "huber"
    assert len(result.history) == 4 and result.context.turn == 4
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "done"
    assert manifest["resumed_from"]
    assert (target / "checkpoints" / "final.pt").exists()


@pytest.mark.slow
def test_open_record_and_predict_on_new_rows(trained):
    record = Path(trained.record)
    ctx = running.open_record(record)
    assert set(ctx.models) == {"net"} and ctx.record is None
    assert not ctx.models["net"].module.training
    opened = kalfa.open(record)
    rows = housing_frame(20, seed=3)
    table = opened.predict(data=rows)
    assert len(table) == 20
    assert {"row", "price", "pred_net"} <= set(table.columns)
    assert numpy.isfinite(table["pred_net"].to_numpy()).all()
    assert (record / "predictions_frame.parquet").exists()
    assert opened.manifest["status"] == "done"
    assert len(opened.history) == 2
    assert any(event["kind"] == "patch" for event in opened.events)
    assert opened.best["series"] == "val/rmse" and opened.best["turn"] in (1, 2)
    assert opened.plots(only="loss_curve") == ["loss_curve"]
    assert (record / "plots" / "loss_curve.png").exists()
    assert set(opened.plots()) == {"loss_curve", "pred_vs_true"}
    assert (record / "plots" / "pred_vs_true.png").exists()


def test_record_properties_read_the_record_files(tmp_path):
    record = kalfa.open(make_record(tmp_path / "alpha", turns=3))
    assert record.manifest["name"] == "alpha"
    assert len(record.updates) == 12 and list(record.updates["update"])[:3] == [1, 2, 3]
    assert {"loss/main", "lr/main", "grad_norm/main"} <= set(record.updates.columns)
    assert record.progress["turn"] == 3 and record.progress["total"] == 10
    assert record.model_summary["net"]["parameters"] == 97
    assert record.best["series"] == "val/rmse" and record.best["turn"] == 3


def test_record_series_steps_and_patches_read_the_record(tmp_path):
    record = kalfa.open(make_record(tmp_path / "alpha", turns=3))
    table = record.series(["val/rmse"])
    assert isinstance(table, pandas.DataFrame) and list(table.columns) == ["turn", "val/rmse"]
    assert list(table["turn"]) == [1, 2, 3]
    assert list(record.series("val/rmse", limit=2)["turn"]) == [1, 3]
    whole = record.series()
    assert len(whole) == 3 and {"train/mse", "val/rmse", "lr/main"} <= set(whole.columns)
    assert not {"update", "seconds", "rules"} & set(whole.columns)
    steps = record.steps(["loss/main"], limit=5)
    assert isinstance(steps, pandas.DataFrame) and list(steps.columns) == ["update", "turn", "loss/main"]
    assert len(steps) == 5 and (steps["update"].iloc[0], steps["update"].iloc[-1]) == (1, 12)
    assert len(record.steps()) == 12
    patches = record.patches()
    assert len(patches) == 1 and patches[0]["kind"] == "patch" and patches[0]["path"] == "optimizers.main.minimizes"


def test_record_state_stats_and_compare_read_the_record(tmp_path):
    first = make_record(tmp_path / "alpha")
    second = make_record(tmp_path / "beta", params={"lr": 0.01}, seed=2)
    record = kalfa.open(first)
    assert record.state == "done"
    assert kalfa.open(make_record(tmp_path / "gamma", status="running")).state == "stale"
    assert kalfa.open(tmp_path / "nowhere").state == "unknown"
    stats = record.stats(where="town == 'ank'", bins=10)
    assert stats["rows"] == 25 and [pair["kind"] for pair in stats["pairs"]] == ["regression"]
    assert stats["pairs"][0]["target"] == "price" and stats["pairs"][0]["stats"]["count"] == 25
    assert len(stats["pairs"][0]["stats"]["pred_hist"]["counts"]) == 10
    assert kalfa.open(make_record(tmp_path / "bare", predictions=False)).stats() is None
    assert len(record.predictions) == 50 and "pred_net" in record.predictions.columns
    compared = record.compare(kalfa.open(second), str(first), keys="val/rmse")
    assert [summary["name"] for summary in compared["records"]] == ["alpha", "beta", "alpha"]
    assert compared["params"]["differing"] == {"lr": [0.001, 0.01, 0.001]}
    assert set(compared["series"]) == {"turn", str(first), str(second)}
    assert list(compared["series"][str(second)]) == ["val/rmse"]
    assert list(compared["diffs"]) == ["alpha vs beta", "alpha vs alpha"]
    assert "+  lr: 0.01" in compared["diffs"]["alpha vs beta"] and compared["diffs"]["alpha vs alpha"] == ""


def test_records_lists_record_objects_in_records_under_order(tmp_path):
    make_record(tmp_path / "old", started="2026-11-07T09:00:00")
    make_record(tmp_path / "nested" / "new", started="2026-11-07T12:00:00")
    found = api.records(tmp_path)
    assert all(isinstance(record, kalfa.Record) for record in found)
    assert [record.manifest["name"] for record in found] == ["new", "old"]
    assert [Path(record.directory) for record in found] == [tmp_path / "nested" / "new", tmp_path / "old"]
    assert api.records(tmp_path / "nowhere") == []
    (tmp_path / "empty").mkdir()
    assert api.records(tmp_path / "empty") == []
    make_sweep(tmp_path / "grid", count=2)
    assert [record.manifest["name"] for record in api.records(tmp_path)] == ["new", "0000", "0001", "old"]


def test_summarize_returns_the_summary_of_the_module(tmp_path):
    first = make_record(tmp_path / "alpha", seed=1)
    second = make_record(tmp_path / "beta", seed=2, turns=5)
    found = kalfa.summarize([first, second], by="val/mse", mode="max", top=1)
    assert isinstance(found, summarizing.Summary)
    assert found.by == "val/mse" and found.mode == "max" and len(found.table) == 1
    assert found.best == summarizing.summarize([first, second], by="val/mse", mode="max").best
    assert api.summarize([first, second]).by == "val/rmse"


@pytest.mark.slow
def test_trained_record_summary_has_status_done_and_best(trained):
    record = kalfa.open(trained.record)
    summary = record.summary
    assert summary["status"] == "done" and summary["turn"] == 2 and summary["total"] == 2
    assert summary["best"]["series"] == "val/rmse" and summary["best"]["turn"] in (1, 2)
    assert numpy.isfinite(summary["best"]["value"])
    assert record.state == "done"


@pytest.mark.slow
def test_trained_record_series_has_one_row_per_turn(trained):
    record = kalfa.open(trained.record)
    table = record.series(["val/rmse"])
    assert list(table.columns) == ["turn", "val/rmse"]
    assert list(table["turn"]) == [1, 2] and len(table) == len(record.history)
    assert numpy.isfinite(table["val/rmse"].to_numpy()).all()


def test_record_config_and_fields_read_the_record(tmp_path):
    record = kalfa.open(make_record(tmp_path / "alpha"))
    assert record.config.startswith("seed: 7\n") and "epochs: 10" in record.config
    assert record.fields is None
    (tmp_path / "alpha" / "fitted").mkdir()
    (tmp_path / "alpha" / "fitted" / "fields.json").write_text(json.dumps({"targets": ["price"]}), encoding="utf-8")
    assert record.fields == {"targets": ["price"]}
    bare = kalfa.open(tmp_path / "nowhere")
    assert bare.config is None and bare.fields is None
