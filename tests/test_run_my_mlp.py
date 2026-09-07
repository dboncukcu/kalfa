"""The minimal example end to end: the twin of config 01 with the model net and an inline optimizer."""

from pathlib import Path

import pytest

import kalfa  # noqa: F401
from conftest import ROOT
from kalfa.api import check, run
from kalfa.config import parse_sets
from kalfa.record import read_history
from kalfa.synthetic import write_housing

CONFIG = str(ROOT / "examples" / "minimal" / "config.yaml")
TWIN = str(ROOT / "configs" / "01_mlp_regression.yaml")


@pytest.fixture
def housing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_housing(tmp_path / "housing.parquet")
    return tmp_path


def test_check_my_mlp(housing):
    prepared = check([CONFIG], parse_sets([]))
    assert prepared.problems == []
    items = prepared.document["flow"]["optimizers"]["params"]["optimizer_items"]
    assert [item["name"] for item in items] == ["net"] and items[0]["models"] == {"net": "net"}
    assert prepared.document["flow"]["training"]["params"]["predicts"] == "net"
    assert prepared.document["blocks"]["net"]["inputs"] == ["x"]


def test_run_my_mlp_matches_01(housing):
    result = run([CONFIG], parse_sets(params=["epochs=3"]), when="fixed")
    record = Path(result.record)
    assert record == Path("runs/housing_fixed")
    history = read_history(record)
    assert [line["turn"] for line in history] == [1, 2, 3]
    assert {"train/loss_mse", "train/loss_huber", "val/rmse", "val/mae", "test/rmse", "lr/net"} <= set(history[0])
    for name in ("checkpoints/best.pt", "checkpoints/last.pt", "final/state.pt", "predictions.parquet",
                 "plots/loss_curve.png", "plots/pred_vs_true.png"):
        assert (record / name).exists(), name
    twin = run([TWIN], parse_sets(params=["epochs=3"]), when="twin")
    other = read_history(Path(twin.record))
    assert [line["val/rmse"] for line in history] == [line["val/rmse"] for line in other]
    assert [line["lr/net"] for line in history] == [line["lr/model"] for line in other]
