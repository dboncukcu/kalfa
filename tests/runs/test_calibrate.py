from pathlib import Path

import pandas
import pytest

from helpers import minimal, write_config
from kalfa.api import check, predict, run
from kalfa.config import parse_sets

pytestmark = pytest.mark.slow


def calibrated():
    config = minimal()
    config["params"]["epochs"] = 2
    config["data"]["mask"] = "x0 > 1.5"
    config["calibrate"] = {"cut": {"uri": "threshold", "params": {"set": "valid", "quantile": 0.9}}}
    config["record"] = "runs/calibrated"
    return config


def test_the_mask_and_the_calibration_reach_the_record_and_predict(workdir):
    path = write_config(workdir / "cfg.yaml", calibrated())
    prepared = check([str(path)], parse_sets([]))
    assert prepared.errors == []
    result = run([str(path)], parse_sets([]))
    record = Path(result.record)
    assert (record / "fitted" / "calibrate" / "calibrations.pkl").exists()
    predictions = pandas.read_parquet(record / "predictions.parquet")
    source = pandas.read_parquet("housing.parquet")
    assert "flag_y" in predictions and (source["x0"].iloc[predictions["row"]] <= 1.5).all()
    again = predict(record)
    assert "flag_y" in again.table and len(again.table) == len(predictions)
    fresh = predict(record, data=source.head(50))
    assert "flag_y" in fresh.table and len(fresh.table) == int((source["x0"].head(50) <= 1.5).sum())
    bad = calibrated()
    bad["calibrate"] = {"cut": {"uri": "adam"}}
    kinds = [problem.kind for problem in check([str(write_config(workdir / "bad.yaml", bad))], parse_sets([])).problems]
    assert "kind_mismatch" in kinds
