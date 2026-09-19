from pathlib import Path

import pytest

from helpers import config_path, copied
from kalfa.api import check, predict
from kalfa.config import parse_sets
from kalfa.std.common.history import History


pytestmark = pytest.mark.slow


@pytest.fixture
def simclr(trained, at_root):
    return trained("simclr")


def test_two_views_names_another_preprocessor_and_stays_on_the_train_set(at_root):
    prepared = check([config_path("simclr")], parse_sets([]))
    assert prepared.problems == []
    data = prepared.document["flow"]["data"]["params"]
    assert data["prep"]["params"]["preprocessors"]["two"] == {
        "uri": "/pre/kalfa/two_views", "params": {"transform": {"uri": "/pre/kalfa/simclr_aug", "params": {"size": 8}}}}
    assert data["preprocessors_keys"] == {"aug": {}, "two": {"sets": ["train"]}, "to_tensor": {}}
    assert prepared.document["flow"]["training"]["params"]["losses_keys"] == {"ntx": {"sets": ["train"]}}
    ghost = check([config_path("simclr")], parse_sets(["data.preprocessors.two.params.transform=ghost"]))
    assert [problem.kind for problem in ghost.errors] == ["unresolved_ref"]


def test_a_run_without_targets_reports_the_train_loss_alone(simclr):
    history = History.read(simclr.record)
    assert [list(line) for line in history] == [["turn", "global_step", "train/ntx", "lr/main", "minimizes/main",
                                                 "seconds", "rules"]]
    assert history[0]["global_step"] == 8
    assert not (Path(simclr.record) / "predictions.parquet").exists()
    assert sorted(path.name for path in (Path(simclr.record) / "checkpoints").iterdir()) == ["last.pt"]


def test_predict_with_the_backbone_embeds_every_image_once(simclr, tmp_path):
    copy = copied(simclr.record, tmp_path / "simclr_copy")
    result = predict(copy, model="backbone", data="images")
    assert result.path == str(copy / "predictions_images_backbone.parquet")
    assert list(result.table.columns) == ["row", "raw_h_0", "raw_h_1", "raw_h_2", "raw_h_3"]
    assert len(result.table) == 64
