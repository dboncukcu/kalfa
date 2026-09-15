from pathlib import Path

import pandas
import pytest

from helpers import minimal, write_config
from kalfa.api import check, predict, run
from kalfa.config import parse_sets

pytestmark = pytest.mark.slow


def framed():
    config = minimal()
    config["params"]["epochs"] = 2
    config["data"]["transform"] = [{"uri": "derive", "params": {"column": "bucket", "expr": "x0 > 0"}}]
    config["data"]["frame"] = [{"uri": "group_statistic", "params": {"by": ["bucket"], "column": "price"}},
                               {"uri": "target_encoding", "params": {"column": "bucket", "target": "price"}}]
    config["data"]["fields"]["price_mean_by_bucket"] = {"preprocessors": ["std_scaler"]}
    config["data"]["fields"]["bucket_target"] = {"preprocessors": ["std_scaler"]}
    config["record"] = "runs/framed"
    return config


def test_frame_transforms_fit_on_train_and_replay_for_new_data(workdir):
    path = write_config(workdir / "cfg.yaml", framed())
    prepared = check([str(path)], parse_sets([]), measure=True)
    assert prepared.errors == []
    result = run([str(path)], parse_sets([]))
    record = Path(result.record)
    assert (record / "fitted" / "frames" / "frames.pkl").exists()
    plan = (record / "fitted" / "preprocessors" / "plan.json").read_text()
    assert "price_mean_by_bucket" in plan and "bucket_target" in plan
    new = pandas.read_parquet("housing.parquet").head(20)
    prediction = predict(record, data=new)
    assert len(prediction.table) == 20 and Path(prediction.path).name == "predictions_frame.parquet"
    stream = framed()
    stream["include"] = ["/alias/kalfa/lazy_tabular"]
    stream["data"]["split"] = {"uri": "sequential", "params": {"ratios": [0.7, 0.15, 0.15]}}
    lazy = check([str(write_config(workdir / "lazy.yaml", stream))], parse_sets([]))
    assert "lazy_frame" in [problem.kind for problem in lazy.problems]
