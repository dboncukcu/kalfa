from pathlib import Path

import pytest

from kalfa.api import check, predict
from kalfa.config import parse_sets
from kalfa.std.common.history import History

pytestmark = pytest.mark.slow


def test_check_reads_the_image_folder(dataset):
    dataset("05_autoencoder")
    prepared = check(["config.yaml"])
    assert prepared.problems == []
    assert prepared.sizes == {"train": 180, "valid": 20, "test": 0}
    prepared = check(["config.yaml"], parse_sets(["data.fields.image.preprocessors=[]"]))
    assert "dtype_unsupported" in [problem.kind for problem in prepared.problems]


def test_composite_predicts_and_the_encoder_codes(trained):
    result = trained("05_autoencoder")
    record = Path(result.record)
    history = History.read(record)
    assert [line["turn"] for line in history] == [1, 2]
    assert {"train/rec", "train/recon_rmse", "val/rec", "val/recon_rmse", "lr/main"} <= set(history[0])
    assert not any(key.startswith("test/") for key in history[0])
    assert history[-1]["train/rec"] < history[0]["train/rec"]
    assert not (record / "predictions.parquet").exists()
    assert (record / "plots" / "reconstructions.png").exists() and (record / "plots" / "loss_curve.png").exists()
    assert (record / "checkpoints" / "best.pt").exists()
    codes = predict(result.record, model="encoder", data="data/mnist_new")
    assert len(codes.table) == 20 and list(codes.table.columns)[:2] == ["row", "raw_z_0"]
    assert len(codes.table.columns) == 33 and codes.model == "encoder"
    own = predict(result.record)
    assert len(own.table) == 0
