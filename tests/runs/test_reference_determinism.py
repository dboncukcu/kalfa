import json
from pathlib import Path

import pandas
import pytest
import torch

from kalfa.std.checkpoint.base import load
from kalfa.std.common.history import History


pytestmark = pytest.mark.slow


def without_seconds(record):
    return [{key: value for key, value in line.items() if key != "seconds"} for line in History.read(record)]


def tensors_equal(first, second):
    assert list(first) == list(second)
    for name in first:
        assert list(first[name]) == list(second[name])
        for key in first[name]:
            assert torch.equal(first[name][key], second[name][key]), (name, key)


def test_the_same_seed_reproduces_the_history_and_the_weights(reference, trained):
    twin = trained("reference", when="twin")
    assert twin.record == "runs/ref_twin"
    assert without_seconds(twin.record) == without_seconds(reference.record)
    assert History.read_steps(twin.record) == History.read_steps(reference.record)
    first = load(Path(reference.record) / "final" / "state.pt")
    second = load(Path(twin.record) / "final" / "state.pt")
    tensors_equal(first["models"], second["models"])
    tensors_equal(first["emas"], second["emas"])
    assert first["counters"] == second["counters"] and first["rules"]["effects"] == second["rules"]["effects"]
    pandas.testing.assert_frame_equal(pandas.read_parquet(Path(twin.record) / "predictions.parquet"),
                                      pandas.read_parquet(Path(reference.record) / "predictions.parquet"))


def test_another_rng_rule_changes_the_weights_but_not_the_data(reference, trained):
    other = trained("reference", when="indexed", sets=["rng=indexed"])
    assert other.record == "runs/ref_indexed"
    assert json.loads((Path(other.record) / "data.json").read_text()) == json.loads(
        (Path(reference.record) / "data.json").read_text())
    first = load(Path(reference.record) / "final" / "state.pt")["models"]
    second = load(Path(other.record) / "final" / "state.pt")["models"]
    assert not torch.equal(first["tower"]["nodes.stem.weight"], second["tower"]["nodes.stem.weight"])
    assert (pandas.read_parquet(Path(other.record) / "predictions.parquet")["row"].to_numpy()
            == pandas.read_parquet(Path(reference.record) / "predictions.parquet")["row"].to_numpy()).all()
