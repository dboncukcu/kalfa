"""Config 01 end to end on the synthetic housing table: the record directory, rules, resume, predict, seeds."""

import json
import warnings
from pathlib import Path

import pandas
import pytest
import torch

import kalfa  # noqa: F401
from kalfa.api import predict, resume, run
from kalfa.config import parse_sets
from kalfa.record import read_history, read_resolved, resume_chain
from kalfa.synthetic import write_housing

FORCED_RULES = ("training.rules=[{name: to_huber, when: {uri: after_epoch, params: {at: 1}}, set: {loss: loss_huber}}, "
                "{name: to_mae, after: to_huber, when: {uri: metric_below, params: {monitor: train/loss_huber, "
                "value: 1e9}}, set: {loss: loss_mae}}, {name: to_logcosh, after: to_mae, when: {uri: plateau, "
                "params: {monitor: val/loss_mae, patience: 5}}, set: {loss: loss_logcosh}}]")


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    directory = tmp_path_factory.mktemp("run01")
    write_housing(directory / "housing.parquet")
    import os

    previous = os.getcwd()
    os.chdir(directory)
    try:
        result = run([str(Path(__file__).resolve().parents[1] / "configs" / "01_mlp_regression.yaml")],
                     parse_sets(params=["epochs=3"]), when="fixed")
    finally:
        os.chdir(previous)
    return directory, result


def test_record_directory_contents(trained):
    directory, result = trained
    record = directory / result.record
    assert record == directory / "runs" / "housing_fixed"
    for name in ("resolved.yaml", "flow.yaml", "history.jsonl", "events.jsonl", "run.json", "stdout.txt",
                 "checkpoints/best.pt", "checkpoints/last.pt", "final/state.pt", "preprocessors/plan.json",
                 "preprocessors/std_scaler.pkl", "preprocessors/target_std.pkl", "predictions.parquet",
                 "plots/loss_curve.png", "plots/pred_vs_true.png"):
        assert (record / name).exists(), name
    resolved = (record / "resolved.yaml").read_text()
    assert "epochs: 3  # --set overrides" in resolved and "record: runs/housing_$datetime$" in resolved
    assert "uri: /source/kalfa/parquet" in resolved and "include" not in resolved
    config = read_resolved(record)
    assert config["training"]["turn"] == "/turn/kalfa/alternating"
    history = read_history(record)
    assert [line["turn"] for line in history] == [1, 2, 3]
    keys = set(history[0])
    assert {"train/loss_mse", "train/loss_huber", "train/loss_mae", "train/loss_logcosh", "train/rmse", "train/mae",
            "val/rmse", "val/mae", "val/loss_mse", "test/rmse", "test/loss_mse", "lr/model", "global_step",
            "rules"} <= keys
    assert history[-1]["val/rmse"] < history[0]["val/rmse"]
    assert list(history[0])[:2] == ["turn", "global_step"] and history[0]["rules"] == []
    predictions = pandas.read_parquet(record / "predictions.parquet")
    assert list(predictions.columns) == ["row", "price", "raw_y", "pred_y"] and len(predictions) == 300
    source = pandas.read_parquet(directory / "housing.parquet")
    assert predictions["price"].to_numpy() == pytest.approx(source["price"].iloc[predictions["row"]].to_numpy(),
                                                            abs=1e-3)
    assert abs(float((predictions["pred_y"] - predictions["price"]).abs().mean())) < 20.0
    payload = torch.load(record / "checkpoints" / "last.pt", weights_only=False)
    assert payload["counters"] == {"global_step": 33, "turn": 3} and payload["checkpoint"]["best"] == min(
        line["val/rmse"] for line in history)
    assert result.report.outputs["history"] == history_without_bookkeeping(history)
    flow = (record / "flow.yaml").read_text()
    assert flow.startswith("components:") and "build_model:" in flow


def history_without_bookkeeping(history):
    return [{key: value for key, value in line.items() if key not in ("turn", "global_step", "rules")
             and not key.startswith("lr/")} for line in history]


def test_forced_rule_chain_fires_and_switches_the_loss(workdir, config_01):
    result = run([config_01], parse_sets([FORCED_RULES], ["epochs=4"]), when="forced")
    history = read_history(result.record)
    assert [line["rules"] for line in history] == [["to_huber"], ["to_mae"], [], []]
    payload = torch.load(Path(result.record) / "checkpoints" / "last.pt", weights_only=False)
    assert payload["rules"]["sticky"] == ["to_huber", "to_mae"] and payload["rules"]["effects"] == {"loss": "loss_mae"}
    assert "best" in payload["rules"]["triggers"]["to_logcosh"] and "to_huber" in payload["rules"]["triggers"]


def test_resume_continues_from_last_pt_into_a_new_directory(trained):
    directory, result = trained
    import os

    previous = os.getcwd()
    os.chdir(directory)
    try:
        continued = resume(result.record, parse_sets(["training.epochs=5"]), when="resumed")
    finally:
        os.chdir(previous)
    record = directory / continued.record
    assert record != directory / result.record and record.name == "housing_resumed"
    history = read_history(record)
    assert [line["turn"] for line in history] == [4, 5] and history[0]["global_step"] == 44
    assert resume_chain(record) == [result.record]
    assert (record / "checkpoints" / "best.pt").exists() and (record / "final" / "state.pt").exists()
    assert json.loads((record / "resume.json").read_text())["checkpoint"].endswith("checkpoints/last.pt")
    with pytest.raises(Exception):
        resume(str(directory / "nowhere"))


def test_predict_on_new_data_inverts_the_target(trained):
    directory, result = trained
    new = write_housing(directory / "new.parquet", rows=40, seed=5)
    import os

    previous = os.getcwd()
    os.chdir(directory)
    try:
        prediction = predict(result.record, data=str(new))
        own = predict(result.record)
    finally:
        os.chdir(previous)
    table = prediction.table
    assert list(table.columns) == ["row", "price", "raw_y", "pred_y"] and len(table) == 40
    truth = pandas.read_parquet(new)["price"].to_numpy()
    assert table["price"].to_numpy() == pytest.approx(truth, abs=1e-3)
    assert abs(float((table["pred_y"].to_numpy() - truth).mean())) < 20.0
    assert Path(prediction.path).name == "predictions_new.parquet" and prediction.model == "model"
    assert len(own.table) == 300 and Path(own.path).name == "predictions.parquet"


def test_same_seed_same_history_and_no_seed_warns(workdir, config_01):
    first = run([config_01], parse_sets(params=["epochs=2"]), when="seed_a")
    second = run([config_01], parse_sets(params=["epochs=2"]), when="seed_b")
    assert read_history(first.record) == read_history(second.record)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run([config_01], parse_sets(["seed=null"], ["epochs=1"]), when="unseeded")
    assert any("no_seed" in str(entry.message) for entry in caught)


def test_existing_record_is_refused(workdir, config_01):
    run([config_01], parse_sets(params=["epochs=1"]), when="twice")
    with pytest.raises(Exception, match="exists"):
        run([config_01], parse_sets(params=["epochs=1"]), when="twice")
