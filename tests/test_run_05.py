"""Config 05 end to end on a tiny image folder: composite predicts, target input, encoder codes in a prediction."""

from pathlib import Path

import pytest

import kalfa  # noqa: F401
from conftest import ROOT
from kalfa.api import check, predict, run
from kalfa.config import parse_sets
from kalfa.record import read_history
from kalfa.synthetic import write_image_folder

CONFIG = str(ROOT / "configs" / "05_autoencoder.yaml")


@pytest.fixture
def mnist(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_image_folder(tmp_path / "data" / "mnist", classes=("zero", "one"), per_class=24, size=28)
    return tmp_path


def test_check_reads_the_image_folder(mnist):
    prepared = check([CONFIG])
    assert prepared.problems == []
    assert prepared.sizes == {"train": 43, "valid": 5, "test": 0}
    prepared = check([CONFIG], parse_sets(["data.fields.image.preprocessors=[]"]))
    assert "dtype_unsupported" in [problem.kind for problem in prepared.problems]


def test_run_05(mnist):
    result = run([CONFIG], parse_sets(["data.batch=16"], ["epochs=2"]), when="fixed")
    record = Path(result.record)
    history = read_history(record)
    assert [line["turn"] for line in history] == [1, 2]
    assert {"train/rec", "train/recon_rmse", "val/rec", "val/recon_rmse", "lr/main"} <= set(history[0])
    assert not any(key.startswith("test/") for key in history[0])
    assert history[-1]["train/rec"] < history[0]["train/rec"]
    assert not (record / "predictions.parquet").exists()
    assert (record / "plots" / "reconstructions.png").exists() and (record / "plots" / "loss_curve.png").exists()
    assert (record / "checkpoints" / "best.pt").exists()
    codes = predict(result.record, model="encoder", data=str(mnist / "data" / "mnist"))
    assert len(codes.table) == 48 and list(codes.table.columns)[:2] == ["row", "raw_z_0"]
    assert len(codes.table.columns) == 33 and codes.model == "encoder"
    own = predict(result.record)
    assert len(own.table) == 0
