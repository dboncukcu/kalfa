import json
import math
from pathlib import Path

import numpy
import pandas
import pytest

import kalfa  # noqa: F401
from conftest import ROOT
from kalfa.api import check, predict, run
from kalfa.config import parse_sets
from kalfa.record import read_history
from kalfa.synthetic import write_scores

CONFIG = str(ROOT / "configs" / "15_multi_target.yaml")


@pytest.fixture
def scores(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_scores(tmp_path / "scores.parquet")
    return tmp_path


def test_check_resolves_the_target_table(scores):
    prepared = check([CONFIG])
    assert prepared.problems == []
    keys = prepared.document["flow"]["training"]["params"]["losses_keys"]
    assert keys["l_y"] == {"output": "y_hat", "target": "y_*"}
    assert keys["l_z"] == {"output": "z_hat", "target": "z"}
    assert keys["total"] == {}
    metrics = prepared.document["flow"]["training"]["params"]["metrics_keys"]
    assert metrics["rmse_y"] == {"output": "y_hat", "target": "y_*"}
    assert prepared.document["flow"]["after"]["params"]["targets"] == {"y_hat": "y_*", "z_hat": "z",
                                                                      "tail_logit": "z_tail"}


def test_run_15(scores):
    result = run([CONFIG], parse_sets(params=["epochs=4"]), when="fixed")
    record = Path(result.record)
    history = read_history(record)
    assert [line["turn"] for line in history] == [1, 2, 3, 4]
    keys = set(history[0])
    assert {"train/l_y", "train/l_z", "train/l_tail", "val/rmse_y", "val/rmse_z", "val/auroc_tail"} <= keys
    assert all(math.isfinite(line["train/l_y"]) for line in history)
    predictions = pandas.read_parquet(record / "predictions.parquet")
    assert list(predictions.columns) == ["row", "y_a", "y_b", "y_c", "z", "z_tail",
                                         "raw_y_hat_0", "raw_y_hat_1", "raw_y_hat_2",
                                         "pred_y_hat_y_a", "pred_y_hat_y_b", "pred_y_hat_y_c",
                                         "raw_z_hat", "pred_z_hat_z", "raw_tail_logit", "pred_tail_logit_z_tail"]
    source = pandas.read_parquet(scores / "scores.parquet")
    for column in ("y_a", "y_b", "y_c", "z"):
        observed = predictions[column].to_numpy()
        assert numpy.allclose(observed, source[column].iloc[predictions["row"]].to_numpy(), atol=1e-4)
    plan = json.loads((record / "preprocessors" / "plan.json").read_text())
    assert [item["name"] for item in plan["fields"] if item["target"]] == ["y_a", "y_b", "y_c", "z", "z_tail"]
    assert (record / "plots" / "pred_vs_true.png").exists()
    prediction = predict(result.record)
    assert "pred_y_hat_y_b" in prediction.table.columns


def test_every_target_keeps_its_own_scale(scores):
    from kalfa.std.pre import fit, standard_scaler

    source = pandas.read_parquet(scores / "scores.parquet")
    prep = fit(source, {"y_*": {"target": True, "preprocessors": ["s"]}}, {"s": standard_scaler()}, [])
    grouped = prep.fitted["s"]
    assert grouped.columns == ["y_a", "y_b", "y_c"]                    # one object, one column of statistics each
    means = [float(grouped.obj.scaler.mean_[position]) for position in range(3)]
    assert len(set(round(value, 6) for value in means)) == 3
    values = prep.inverse("y_b", [0.0, 1.0])
    assert abs(values[0] - means[1]) < 1e-6
