from pathlib import Path

import pandas
import pytest

from kalfa.api import check
from kalfa.config import parse_sets
from kalfa.std.common.history import History

pytestmark = pytest.mark.slow


def test_check_column_refs(dataset):
    dataset("03_timeseries_window")
    prepared = check(["config.yaml"])
    assert prepared.problems == []
    assert prepared.sizes == {"train": 1260, "valid": 270, "test": 270}
    prepared = check(["config.yaml"], parse_sets(["data.drop=[site_id]"]))
    assert "dropped_column_ref" in [problem.kind for problem in prepared.problems]
    prepared = check(["config.yaml"], parse_sets(["data.fields.site_id={}"]))
    assert "column_in_fields" in [problem.kind for problem in prepared.problems]


def test_windows_with_context_and_the_forecast(trained):
    result = trained("03_timeseries_window")
    record = Path(result.record)
    history = History.read(record)
    assert [line["turn"] for line in history] == [1, 2]
    assert {"train/loss_mse", "train/rmse", "val/rmse", "val/mae", "test/rmse", "lr/model"} <= set(history[0])
    assert history[-1]["val/rmse"] > 2.0 and history[-1]["train/loss_mse"] < 2.0
    predictions = pandas.read_parquet(record / "predictions.parquet")
    assert list(predictions.columns)[:2] == ["row", "load_0"]
    predicted = [column for column in predictions.columns if column.startswith("pred_")]
    assert predicted == [f"pred_y_{i}" for i in range(24)]
    assert len(predictions) == 3 * (90 - 24 + 1)
    source = pandas.read_parquet("energy.parquet")
    first = predictions.iloc[0]
    assert first["load_0"] == pytest.approx(source.loc[int(first["row"]), "load"], abs=1e-2)
    assert first["load_23"] == pytest.approx(source.loc[int(first["row"]) + 23, "load"], abs=1e-2)
    assert (record / "plots" / "forecast.png").exists()
    assert abs(float(predictions["pred_y_0"].mean()) - float(predictions["load_0"].mean())) < 60.0
