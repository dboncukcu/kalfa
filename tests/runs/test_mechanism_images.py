import math
from pathlib import Path

import pandas
import pytest
import torch
from torch.utils.data import WeightedRandomSampler

from helpers import config_path, copied
from kalfa.api import check, open_record, predict, record_loaders
from kalfa.config import parse_sets
from kalfa.std.checkpoint.base import load
from kalfa.std.common.history import History


pytestmark = pytest.mark.slow

KEYS = ["turn", "global_step", "train/ce", "train/accuracy", "train/f1", "val/ce", "val/accuracy", "val/f1",
        "test/ce", "test/accuracy", "test/f1", "lr/main", "minimizes/main", "seconds", "rules"]


@pytest.fixture
def images(trained, at_root):
    return trained("images")


def test_the_per_item_chain_and_the_balanced_sampler_are_declared_to_the_data_block(at_root):
    prepared = check([config_path("images")], parse_sets([]))
    assert prepared.problems == [] and prepared.sizes == {"train": 38, "valid": 13, "test": 13}
    params = prepared.document["flow"]["data"]["params"]
    assert params["preprocessors_keys"] == {"resize": {}, "augment": {"sets": ["train"]}, "to_tensor": {},
                                            "normalize": {}}
    assert params["loaders"]["train"]["params"]["balanced"] is True
    assert params["loaders"]["valid"]["params"]["balanced"] is True


def test_the_train_loader_draws_with_a_weighted_sampler_that_lifts_the_minority(images):
    opened = open_record(images.record)
    loaders = record_loaders(opened.document, opened.contract)
    assert isinstance(loaders["train"].sampler, WeightedRandomSampler)
    assert loaders["valid"].sampler.__class__.__name__ == "SequentialSampler"
    labels = torch.cat([batch["label"].reshape(-1) for batch in loaders["train"]])
    assert len(labels) == 38 and 0.3 < float((labels == 1).float().mean()) < 0.7


def test_the_history_and_the_checkpoint_follow_the_classification_run(images):
    history = History.read(images.record)
    assert [list(line) for line in history] == [KEYS]
    assert history[0]["global_step"] == math.ceil(38 / 8)
    best = load(Path(images.record) / "checkpoints" / "best.pt")
    assert best["checkpoint"] == {"best": history[0]["val/accuracy"]} and best["turn"] == 1
    assert best["models"]["net"]["nodes.s0.weight"].shape == (4, 1, 3, 3)
    assert best["models"]["net"]["nodes.s4.weight"].shape == (2, 36)


def test_the_predictions_carry_the_decoded_class_of_every_test_image(images):
    table = pandas.read_parquet(Path(images.record) / "predictions.parquet")
    assert list(table.columns) == ["row", "label", "raw_logits_0", "raw_logits_1"]
    assert len(table) == 13 and set(table["label"]) <= {0, 1}
    plots = sorted(path.name for path in (Path(images.record) / "plots").iterdir())
    assert plots == ["loss_curve.png"]


def test_predict_on_a_folder_scores_every_image(images, tmp_path):
    copy = copied(images.record, tmp_path / "images_copy")
    result = predict(copy, data="images")
    assert result.path == str(copy / "predictions_images.parquet")
    assert list(result.table.columns) == ["row", "label", "raw_logits_0", "raw_logits_1"]
    assert len(result.table) == 64 and result.table["label"].value_counts().to_dict() == {0: 40, 1: 24}
