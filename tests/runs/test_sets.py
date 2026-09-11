from pathlib import Path

import numpy
import pandas
import pytest

import kalfa
from helpers import minimal, write_config
from kalfa.api import check, predict, run
from kalfa.config import parse_sets
from kalfa.std.common.history import History

pytestmark = pytest.mark.slow


def four_way(df, ratios, seed=0):
    order = numpy.random.default_rng(seed).permutation(len(df))
    edges = numpy.cumsum([int(round(ratio * len(df))) for ratio in ratios[:-1]])
    parts = numpy.split(order, edges)
    return {name: df.iloc[numpy.sort(part)] for name, part in zip(("train", "valid", "test", "calib"), parts)}


def test_a_split_with_four_sets_reaches_the_history_the_predictions_and_the_calibration(workdir):
    kalfa.lego("/split/test/four", four_way, returns=["train", "valid", "test", "calib"])
    config = minimal()
    config["params"]["epochs"] = 2
    config["data"]["split"] = {"uri": "/split/test/four", "params": {"ratios": [0.6, 0.15, 0.15, 0.1]}}
    config["metrics"]["rmse"] = {"uri": "rmse", "sets": ["valid", "calib"]}
    config["calibrate"] = {"cut": {"uri": "threshold", "params": {"set": "calib", "quantile": 0.5}}}
    config["record"] = "runs/four"
    path = write_config(workdir / "cfg.yaml", config)
    prepared = check([str(path)], parse_sets([]), measure=True)
    assert prepared.errors == [] and prepared.sets == ["train", "valid", "test", "calib"]
    assert prepared.measured["calib"] == 200 and prepared.document["flow"]["training"]["params"]["sets"] == \
        ["valid", "test", "calib"]
    result = run([str(path)], parse_sets([]))
    record = Path(result.record)
    history = History.read(record)
    assert "calib/rmse" in history[0] and "val/rmse" in history[0] and "test/rmse" not in history[0]
    extra = pandas.read_parquet(record / "predictions_calib.parquet")
    assert len(extra) == 200 and "flag_y" in extra and len(pandas.read_parquet(record / "predictions.parquet")) == 300
    again = predict(record)
    assert "flag_y" in again.table
    wrong = dict(config)
    wrong["metrics"] = {"rmse": {"uri": "rmse", "sets": ["holdout"]}}
    problems = check([str(write_config(workdir / "wrong.yaml", wrong))], parse_sets([])).problems
    assert any(problem.kind == "invalid_value" and "holdout" in problem.message or "sets must list" in problem.message
               for problem in problems)
