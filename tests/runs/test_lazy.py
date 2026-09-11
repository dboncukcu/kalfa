from pathlib import Path

import pandas
import pytest

from helpers import minimal, write_config, write_housing
from kalfa.api import check, predict, run
from kalfa.config import parse_sets
from kalfa.std.common.history import History

pytestmark = pytest.mark.slow


def lazy_config(epochs=2):
    config = minimal()
    config["params"]["epochs"] = epochs
    config["include"] = ["/alias/kalfa/lazy"]
    config["data"]["source"] = {"uri": "parquet", "params": {"path": "housing.parquet", "chunk": 300}}
    config["data"]["split"] = {"uri": "sequential", "params": {"ratios": [0.7, 0.15, 0.15]}}
    config["data"]["transform"] = ["price > 0"]
    config["record"] = "runs/lazy_$datetime$"
    return config


def test_check_refuses_what_the_lazy_set_cannot_do(workdir):
    path = write_config(workdir / "cfg.yaml", lazy_config())
    prepared = check([str(path)], parse_sets([]), load=True)
    assert prepared.problems == [] and prepared.sizes == {"train": 1400, "valid": 300, "test": 300}
    assert prepared.loaded == {"train": 1400, "valid": 300, "test": 300}
    shuffled = lazy_config()
    shuffled["data"]["split"] = {"ratios": [0.7, 0.15, 0.15], "seed": 1}
    shuffled["data"]["batch"] = {"size": 64, "balanced": True}
    shuffled["data"]["feed"] = {"uri": "/feed/kalfa/window", "params": {"size": 2, "horizon": 1}}
    other = write_config(workdir / "shuffled.yaml", shuffled)
    kinds = [problem.kind for problem in check([str(other)], parse_sets([])).problems]
    assert {"lazy_split", "lazy_batch", "lazy_feed"} <= set(kinds)
    collecting = lazy_config()
    collecting["data"]["preprocessors"]["enc"] = {"uri": "label_encoder"}
    del collecting["data"]["preprocessors"]["target_std"]
    collecting["data"]["fields"]["price"] = {"target": True, "preprocessors": ["enc"]}
    problems = check([str(write_config(workdir / "hot.yaml", collecting))], parse_sets([])).problems
    assert [problem.kind for problem in problems] == ["lazy_fit"] and problems[0].severity == "warning"


def test_run_on_the_stream(workdir):
    path = write_config(workdir / "cfg.yaml", lazy_config())
    result = run([str(path)], parse_sets([]), when="fixed")
    record = Path(result.record)
    history = History.read(record)
    assert [line["turn"] for line in history] == [1, 2]
    assert {"train/loss_mse", "val/rmse", "test/rmse", "lr/net"} <= set(history[0])
    table = pandas.read_parquet(record / "predictions.parquet")
    assert table["row"].tolist() == list(range(1700, 2000))
    assert list(table.columns) == ["row", "price", "raw_y", "pred_y"]
    assert (record / "fitted" / "preprocessors" / "std_scaler.pkl").exists()
    assert (record / "plots" / "loss_curve.png").exists()
    write_housing(workdir / "new.parquet", rows=50, seed=9)
    fresh = predict(result.record, data="new.parquet")
    assert len(fresh.table) == 50 and fresh.table["row"].tolist() == list(range(50))
