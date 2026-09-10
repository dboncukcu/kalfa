from pathlib import Path

import pytest
import torch

from kalfa.api import check, prepare, run
from kalfa.config import parse_sets
from kalfa.std.common.history import History
from runs.conftest import SMALL

pytestmark = pytest.mark.slow

SETS, PARAMS = SMALL["04_cnn_images"]


def test_check_reads_the_image_folder(dataset):
    dataset("04_cnn_images")
    prepared = check(["config.yaml"], parse_sets(SETS))
    assert prepared.problems == []
    assert prepared.sizes == {"train": 130, "valid": 14, "test": 0}
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


def test_frozen_backbone_then_unfreeze(dataset, trained):
    dataset("04_cnn_images")
    frozen = run(["config.yaml"], parse_sets(SETS, ["epochs=1", "unfreeze_at=5"]), when="frozen")
    history = History.read(frozen.record)
    assert {"train/ce", "train/accuracy", "val/accuracy", "lr/main"} <= set(history[0])
    assert history[0]["lr/main"] == pytest.approx(1e-3) and history[0]["rules"] == []
    assert torch.equal(backbone_norm(frozen.record), torch.zeros(8))
    thawed = trained("04_cnn_images")
    history = History.read(thawed.record)
    assert [line["rules"] for line in history] == [["unfreeze"], []]
    assert not torch.equal(backbone_norm(thawed.record), torch.zeros(8))
    payload = torch.load(Path(thawed.record) / "checkpoints" / "last.pt", weights_only=False)
    assert payload["rules"]["effects"] == {"backbone.trainable": True}
    groups = payload["optimizers"]["main"]["torch"]["param_groups"]
    assert [group["lr"] for group in groups] == [1e-3, 1e-5]
    assert (Path(thawed.record) / "plots" / "loss_curve.png").exists()
    assert not (Path(thawed.record) / "predictions.parquet").exists()


def test_the_balanced_sampler_is_wired(dataset):
    dataset("04_cnn_images")
    prepared = prepare(["config.yaml"], parse_sets(SETS), dry=False)
    assert prepared.errors == []
    loaders = prepared.document["flow"]["data"]["params"]["loaders"]
    assert loaders["train"]["params"] == {"set": "train", "size": 16, "eval_size": 32, "balanced": True}
