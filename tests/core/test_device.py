import contextlib

import pytest
import torch

from kalfa.core.device import Device
from kalfa.core.errors import RunError


def test_pick_explicit_specs():
    assert Device.pick("cpu") == torch.device("cpu")
    assert Device.pick(" CPU ").type == "cpu"
    picked = Device.pick("cuda:1")
    assert picked.type == "cuda" and picked.index == 1
    assert Device.pick(2) == torch.device("cuda", 2)
    assert Device.pick(torch.device("cpu")) == torch.device("cpu")
    assert Device.pick(Device("cpu")) == torch.device("cpu")
    assert Device.pick(None).type in ("cpu", "cuda", "mps")


def test_pick_auto_follows_availability():
    picked = Device.pick("auto")
    if torch.cuda.is_available():
        assert picked.type == "cuda"
    elif torch.backends.mps.is_available():
        assert picked.type == "mps"
    else:
        assert picked.type == "cpu"
    assert Device().torch == picked


def test_pick_rejects_nonsense():
    with pytest.raises(RunError):
        Device.pick("abacus")


def test_move_moves_tensors_and_leaves_the_rest():
    device = Device("cpu")
    batch = {"x": torch.ones(2, 3), "label": torch.tensor([0, 1]), "row": [1, 2], "name": "a",
             "pair": (torch.zeros(1), 3), "inner": {"y": torch.zeros(2)}}
    moved = device.move(batch)
    assert set(moved) == set(batch) and moved is not batch
    assert moved["x"].device.type == "cpu" and torch.equal(moved["x"], batch["x"])
    assert moved["label"].dtype == torch.int64
    assert moved["row"] == [1, 2] and moved["name"] == "a"
    assert isinstance(moved["pair"], tuple) and moved["pair"][1] == 3 and moved["pair"][0].device.type == "cpu"
    assert moved["inner"]["y"].device.type == "cpu"
    assert device.move(torch.ones(1)).device.type == "cpu"
    assert device.move(5) == 5 and device.move(None) is None


def test_scaler_autocast_note_and_str():
    device = Device("cpu")
    assert device.scaler(True) is None and device.scaler(False) is None
    assert isinstance(device.autocast(True), contextlib.nullcontext)
    assert isinstance(device.autocast(False), contextlib.nullcontext)
    assert device.note() == {"type": "cpu", "name": None, "spec": "cpu"}
    assert str(device) == "cpu"
    automatic = Device("auto")
    assert automatic.note()["spec"] == "auto" and automatic.note()["type"] == automatic.type
    if automatic.type != "cuda":
        assert automatic.scaler(True) is None
