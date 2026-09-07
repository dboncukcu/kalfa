"""Config 04 end to end on a tiny pets folder: frozen backbone, unfreeze rule, balanced sampler, group learning rate."""

from pathlib import Path

import pytest
import torch

import kalfa  # noqa: F401
from conftest import ROOT
from kalfa.api import check, run
from kalfa.config import parse_sets
from kalfa.record import read_history
from kalfa.synthetic import write_image_folder

CONFIG = str(ROOT / "configs" / "04_cnn_images.yaml")
SMALL = ["device=cpu", "data.batch.size=16", "data.batch.eval_size=32",
         "data.preprocessors.resize={uri: resize, params: {size: 32}}",
         "data.preprocessors.augment={uri: random_crop_flip, params: {size: 32}, sets: [train]}"]


@pytest.fixture
def pets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_image_folder(tmp_path / "data" / "pets", classes=("bird", "cat", "dog"), per_class=12, size=32, channels=3)
    return tmp_path


def test_check_04(pets):
    prepared = check([CONFIG], parse_sets(SMALL))
    assert prepared.problems == []
    assert prepared.sizes == {"train": 32, "valid": 4, "test": 0}
    keys = prepared.document["flow"]["data"]["params"]["preprocessors_keys"]
    assert keys == {"resize": {}, "to_tensor": {}, "augment": {"sets": ["train"]}, "normalize": {}}
    items = prepared.document["flow"]["models"]["params"]["trained_items"]
    assert items[0] == {"name": "backbone", "index": 0, "init": None, "trainable": False, "weights": None}
    optimizer = prepared.document["flow"]["optimizers"]["params"]["optimizer_items"][0]
    assert optimizer["params"]["groups"] == [{"match": "backbone.*", "lr": 1e-05}]
    assert optimizer["models"] == {"backbone": "backbone", "head": "head"}


def backbone_norm(record, tag="last"):
    payload = torch.load(Path(record) / "checkpoints" / f"{tag}.pt", weights_only=False)
    state = payload["models"]["backbone"]
    key = next(name for name in state if name.endswith("norm.running_mean"))
    return state[key]


def test_frozen_backbone_then_unfreeze(pets):
    frozen = run([CONFIG], parse_sets(SMALL, ["epochs=1", "unfreeze_at=5"]), when="frozen")
    history = read_history(frozen.record)
    assert {"train/ce", "train/accuracy", "val/accuracy", "lr/main"} <= set(history[0])
    assert history[0]["lr/main"] == pytest.approx(1e-3) and history[0]["rules"] == []
    assert torch.equal(backbone_norm(frozen.record), torch.zeros(8))
    thawed = run([CONFIG], parse_sets(SMALL, ["epochs=2", "unfreeze_at=1"]), when="thawed")
    history = read_history(thawed.record)
    assert [line["rules"] for line in history] == [["unfreeze"], []]
    assert not torch.equal(backbone_norm(thawed.record), torch.zeros(8))
    payload = torch.load(Path(thawed.record) / "checkpoints" / "last.pt", weights_only=False)
    assert payload["rules"]["effects"] == {"backbone.trainable": True}
    groups = payload["optimizers"]["main"]["torch"]["param_groups"]
    assert [group["lr"] for group in groups] == [1e-3, 1e-5]
    assert (Path(thawed.record) / "plots" / "loss_curve.png").exists()
    assert not (Path(thawed.record) / "predictions.parquet").exists()


def test_balanced_loader_sees_every_class(pets):
    from kalfa.api import prepare

    prepared = prepare([CONFIG], parse_sets(SMALL), dry=False)
    assert prepared.errors == []
    document = prepared.document
    assert document["flow"]["data"]["params"]["batch"] == {"size": 16, "eval_size": 32, "balanced": True}
