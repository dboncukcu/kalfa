from pathlib import Path

import pytest

from kalfa.api import check, predict
from kalfa.config import parse_sets
from kalfa.record import read_history
from runs.conftest import SMALL

pytestmark = pytest.mark.slow

SETS, PARAMS = SMALL["09_simclr"]


def test_check_resolves_the_view_preprocessor(dataset):
    dataset("09_simclr")
    prepared = check(["config.yaml"], parse_sets(SETS))
    assert prepared.problems == []
    document = prepared.document
    two = document["flow"]["data"]["params"]["preprocessors"]["two_views"]
    assert two == {"uri": "/pre/kalfa/two_views", "params": {"transform": {"uri": "/pre/kalfa/simclr_aug",
                                                                             "params": {"size": 32}}}}
    assert document["flow"]["data"]["params"]["preprocessors_keys"]["two_views"] == {"sets": ["train"]}
    assert document["flow"]["training"]["params"]["losses_keys"] == {"ntx": {"sets": ["train"]}}
    prepared = check(["config.yaml"], parse_sets(SETS + ["data.preprocessors.two_views.params.transform=ghost"]))
    assert "unresolved_ref" in [problem.kind for problem in prepared.problems]


def test_contrastive_training_and_the_backbone_codes(trained):
    result = trained("09_simclr")
    record = Path(result.record)
    history = read_history(record)
    assert [line["turn"] for line in history] == [1, 2]
    assert set(history[0]) == {"turn", "global_step", "train/ntx", "lr/main", "rules"}
    assert [line["global_step"] for line in history] == [32, 64]
    assert (record / "checkpoints" / "last.pt").exists()
    codes = predict(result.record, model="backbone", data="data/stl10/test")
    assert len(codes.table) == 32 and list(codes.table.columns) == ["row", "raw_h_0", "raw_h_1", "raw_h_2", "raw_h_3"]
    assert codes.model == "backbone"
