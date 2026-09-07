"""Config 09 end to end: two views in the train set only, the ntxent objective, predict with the backbone on new data."""

from pathlib import Path

import pytest
import torch

import kalfa  # noqa: F401
from conftest import ROOT
from kalfa.api import check, predict, run
from kalfa.config import parse_sets
from kalfa.record import read_history
from kalfa.synthetic import write_image_folder

CONFIG = str(ROOT / "configs" / "09_simclr.yaml")
SMALL = ["device=cpu", "data.batch.size=8", "data.preprocessors.simclr_aug.params.size=32"]


@pytest.fixture
def stl(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_image_folder(tmp_path / "data" / "stl10" / "unlabeled", classes=("all",), per_class=16, size=32, channels=3)
    write_image_folder(tmp_path / "data" / "stl10" / "test", classes=("bird", "cat"), per_class=4, size=32, channels=3)
    return tmp_path


def test_check_09(stl):
    prepared = check([CONFIG], parse_sets(SMALL))
    assert prepared.problems == []
    document = prepared.document
    two = document["flow"]["data"]["params"]["preprocessors"]["two_views"]
    assert two == {"uri": "/pre/kalfa/two_views", "params": {"transform": {"uri": "/pre/kalfa/simclr_aug",
                                                                             "params": {"size": 32}}}}
    assert document["flow"]["data"]["params"]["preprocessors_keys"]["two_views"] == {"sets": ["train"]}
    assert document["flow"]["training"]["params"]["losses_keys"] == {"ntx": {"sets": ["train"]}}
    prepared = check([CONFIG], parse_sets(SMALL + ["data.preprocessors.two_views.params.transform=ghost"]))
    assert "unresolved_ref" in [problem.kind for problem in prepared.problems]


def test_run_09(stl):
    result = run([CONFIG], parse_sets(SMALL, ["epochs=2"]), when="fixed")
    record = Path(result.record)
    history = read_history(record)
    assert [line["turn"] for line in history] == [1, 2]
    assert set(history[0]) == {"turn", "global_step", "train/ntx", "lr/main", "rules"}
    assert [line["global_step"] for line in history] == [2, 4]
    assert (record / "checkpoints" / "last.pt").exists()
    codes = predict(result.record, model="backbone", data=str(stl / "data" / "stl10" / "test"))
    assert len(codes.table) == 8 and list(codes.table.columns) == ["row", "raw_h_0", "raw_h_1", "raw_h_2", "raw_h_3"]
    assert codes.model == "backbone"
