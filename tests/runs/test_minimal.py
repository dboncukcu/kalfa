from pathlib import Path

import pytest

from kalfa.api import check
from kalfa.std.common.history import History

pytestmark = pytest.mark.slow


def test_check_the_inline_optimizer(dataset):
    dataset("minimal")
    prepared = check(["config.yaml"])
    assert prepared.problems == []
    items = prepared.document["flow"]["optimizers"]["params"]["optimizer_items"]
    assert [item["name"] for item in items] == ["net"] and items[0]["models"] == {"net": "net"}
    assert prepared.document["flow"]["training"]["params"]["predicts"] == "net"
    assert prepared.document["blocks"]["net"]["inputs"] == ["x"]


def test_the_minimal_run_matches_the_regression_example(trained):
    result = trained("minimal")
    record = Path(result.record)
    assert record == Path("runs/housing_fixed")
    history = History.read(record)
    assert [line["turn"] for line in history] == [1, 2, 3]
    assert {"train/loss_mse", "train/loss_huber", "val/rmse", "val/mae", "test/rmse", "lr/net"} <= set(history[0])
    for name in ("checkpoints/best.pt", "checkpoints/last.pt", "final/state.pt", "predictions.parquet",
                 "plots/loss_curve.png", "plots/pred_vs_true.png"):
        assert (record / name).exists(), name
    twin = trained("01_mlp_regression")
    other = History.read(Path(twin.record))
    assert [line["val/rmse"] for line in history] == [line["val/rmse"] for line in other]
    assert [line["lr/net"] for line in history] == [line["lr/model"] for line in other]
