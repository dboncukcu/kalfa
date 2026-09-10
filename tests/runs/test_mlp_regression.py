import json
import warnings
from pathlib import Path

import pandas
import pytest
import torch

from kalfa.api import check, plots, predict, resume, run
from kalfa.config import parse_sets
from kalfa.record import read_resolved, resume_chain
from kalfa.std.common.history import History

pytestmark = pytest.mark.slow

FORCED_RULES = ("training.rules=[{name: to_huber, when: {uri: after_epoch, params: {at: 1}}, set: {loss: loss_huber}}, "
                "{name: to_mae, after: to_huber, when: {uri: metric_below, params: {monitor: train/loss_huber, "
                "value: 1e9}}, set: {loss: loss_mae}}, {name: to_logcosh, after: to_mae, when: {uri: plateau, "
                "params: {monitor: val/loss_mae, patience: 5}}, set: {loss: loss_logcosh}}]")


def history_without_bookkeeping(history):
    return [{key: value for key, value in line.items() if key not in ("turn", "global_step", "rules", "seconds")
             and not key.startswith("lr/")} for line in history]


def test_record_directory_contents(trained):
    result = trained("01_mlp_regression")
    record = Path(result.record)
    assert record == Path("runs/housing_fixed")
    for name in ("resolved.yaml", "flow.yaml", "history.jsonl", "events.jsonl", "run.json", "stdout.txt",
                 "checkpoints/best.pt", "checkpoints/last.pt", "final/state.pt", "fitted/preprocessors/plan.json",
                 "fitted/preprocessors/std_scaler.pkl", "fitted/preprocessors/target_std.pkl",
                 "fitted/frames/frames.pkl", "predictions.parquet", "data.json",
                 "plots/loss_curve.png", "plots/pred_vs_true.png", "plots/data_pipeline.png"):
        assert (record / name).exists(), name
    resolved = (record / "resolved.yaml").read_text()
    assert "epochs: 3  # --set overrides" in resolved and "record: runs/housing_$datetime$" in resolved
    assert "uri: /source/kalfa/parquet" in resolved and "include" not in resolved
    config = read_resolved(record)
    assert config["training"]["turn"] == "/turn/kalfa/alternating"
    history = History.read(record)
    assert [line["turn"] for line in history] == [1, 2, 3]
    keys = set(history[0])
    assert {"train/loss_mse", "train/loss_huber", "train/loss_mae", "train/loss_logcosh", "train/rmse", "train/mae",
            "val/rmse", "val/mae", "val/loss_mse", "test/rmse", "test/loss_mse", "lr/model", "global_step",
            "rules"} <= keys
    assert history[-1]["val/rmse"] < history[0]["val/rmse"]
    assert list(history[0])[:2] == ["turn", "global_step"] and history[0]["rules"] == []
    predictions = pandas.read_parquet(record / "predictions.parquet")
    assert list(predictions.columns) == ["row", "price", "raw_y", "pred_y"] and len(predictions) == 300
    source = pandas.read_parquet("housing.parquet")
    assert predictions["price"].to_numpy() == pytest.approx(source["price"].iloc[predictions["row"]].to_numpy(),
                                                            abs=1e-3)
    assert abs(float((predictions["pred_y"] - predictions["price"]).abs().mean())) < 20.0
    payload = torch.load(record / "checkpoints" / "last.pt", weights_only=False)
    assert payload["counters"] == {"global_step": 33, "turn": 3} and payload["checkpoint"]["best"] == min(
        line["val/rmse"] for line in history)
    assert result.report.outputs["history"] == history_without_bookkeeping(history)
    flow = (record / "flow.yaml").read_text()
    assert flow.startswith("components:") and "build_model:" in flow


def test_forced_rule_chain_fires_and_switches_the_loss(dataset):
    dataset("01_mlp_regression")
    result = run(["config.yaml"], parse_sets([FORCED_RULES], ["epochs=4"]), when="forced")
    history = History.read(result.record)
    assert [line["rules"] for line in history] == [["to_huber"], ["to_mae"], [], []]
    payload = torch.load(Path(result.record) / "checkpoints" / "last.pt", weights_only=False)
    assert payload["rules"]["sticky"] == ["to_huber", "to_mae"] and payload["rules"]["effects"] == {"loss": "loss_mae"}
    assert "best" in payload["rules"]["triggers"]["to_logcosh"] and "to_huber" in payload["rules"]["triggers"]


def test_resume_continues_from_last_pt_into_a_new_directory(trained):
    result = trained("01_mlp_regression")
    continued = resume(result.record, parse_sets(["training.epochs=5"]), when="resumed")
    record = Path(continued.record)
    assert record != Path(result.record) and record.name == "housing_resumed"
    history = History.read(record)
    assert [line["turn"] for line in history] == [4, 5] and history[0]["global_step"] == 44
    assert resume_chain(record) == [result.record]
    assert (record / "checkpoints" / "best.pt").exists() and (record / "final" / "state.pt").exists()
    assert json.loads((record / "resume.json").read_text())["checkpoint"].endswith("checkpoints/last.pt")
    with pytest.raises(Exception):
        resume("runs/nowhere")


def test_predict_on_new_data_inverts_the_target(trained):
    result = trained("01_mlp_regression")
    prediction = predict(result.record, data="new.parquet")
    own = predict(result.record)
    table = prediction.table
    assert list(table.columns) == ["row", "price", "raw_y", "pred_y"] and len(table) == 100
    truth = pandas.read_parquet("new.parquet")["price"].to_numpy()
    assert table["price"].to_numpy() == pytest.approx(truth, abs=1e-3)
    assert abs(float((table["pred_y"].to_numpy() - truth).mean())) < 20.0
    assert Path(prediction.path).name == "predictions_new.parquet" and prediction.model == "model"
    assert len(own.table) == 300 and Path(own.path).name == "predictions.parquet"


def test_plots_are_redrawn_from_the_record_and_after_a_prediction(trained):
    result = trained("01_mlp_regression")
    record = Path(result.record)
    (record / "plots" / "loss_curve.png").unlink()
    drawn = plots(record, only=["loss_curve"])
    assert drawn.names == ["loss_curve"] and (record / "plots" / "loss_curve.png").exists()
    prediction = predict(record, data="new.parquet", plots="all")
    assert prediction.plots == ["loss_curve", "pred_vs_true", "data_pipeline"]
    assert not (record / "plots" / "data_pipeline_new.png").exists()
    assert (record / "plots" / "pred_vs_true_new.png").exists() and (record / "plots" / "loss_curve_new.png").exists()
    frame = predict(record, data=pandas.read_parquet("new.parquet"))
    assert Path(frame.path).name == "predictions_frame.parquet" and len(frame.table) == 100
    opened = check([str(record)])
    assert opened.errors == [] and opened.surface.paths == [str(record / "resolved.yaml")]
    mapping = check([read_resolved(record)])
    assert mapping.errors == [] and mapping.surface.paths == ["<mapping>"]


def test_same_seed_same_history_and_no_seed_warns(dataset):
    dataset("01_mlp_regression")
    first = run(["config.yaml"], parse_sets(params=["epochs=2"]), when="seed_a")
    second = run(["config.yaml"], parse_sets(params=["epochs=2"]), when="seed_b")
    assert history_without_bookkeeping(History.read(first.record)) == \
        history_without_bookkeeping(History.read(second.record))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run(["config.yaml"], parse_sets(["seed=null"], ["epochs=1"]), when="unseeded")
    assert any("no_seed" in str(entry.message) for entry in caught)


def test_existing_record_is_refused(dataset):
    dataset("01_mlp_regression")
    run(["config.yaml"], parse_sets(params=["epochs=1"]), when="twice")
    with pytest.raises(Exception, match="exists"):
        run(["config.yaml"], parse_sets(params=["epochs=1"]), when="twice")
