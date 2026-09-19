import contextlib

import pytest
import torch
from cirak.registry import registry
from torch import nn

from helpers import build
from kalfa.std import STD_URIS
from kalfa.std.common.device import Device


DEVICE_URIS = sorted(uri for uri in STD_URIS if uri.startswith("/device/"))
CUDA_MESSAGE = "cuda is not available on this machine; write device: cpu, mps or auto"
MPS_MESSAGE = "mps is not available on this machine; write device: cpu, cuda or auto"


def pretend(monkeypatch, cuda=False, mps=False, count=1):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


def test_the_device_pack_holds_auto_and_the_three_devices():
    assert DEVICE_URIS == ["/device/kalfa/auto", "/device/kalfa/cpu", "/device/kalfa/cuda", "/device/kalfa/mps"]


@pytest.mark.parametrize("uri", DEVICE_URIS, ids=[uri.rsplit("/", 1)[1] for uri in DEVICE_URIS])
def test_device_alias_is_its_name(uri):
    assert registry.aliases()[uri.rsplit("/", 1)[1]] == uri


def test_cpu_is_always_there():
    device = build("/device/kalfa/cpu")
    assert device == torch.device("cpu") and device.type == "cpu"


def test_cuda_is_the_indexed_device_when_available():
    if not torch.cuda.is_available():
        pytest.skip("no cuda device on this machine")
    assert build("/device/kalfa/cuda") == torch.device("cuda:0")
    assert build("/device/kalfa/cuda", index=0) == torch.device("cuda:0")
    count = torch.cuda.device_count()
    with pytest.raises(RuntimeError) as caught:
        build("/device/kalfa/cuda", index=count)
    assert str(caught.value) == f"cuda device index {count} does not exist; this machine has {count} cuda device(s)"


def test_cuda_refuses_a_machine_without_it(monkeypatch):
    pretend(monkeypatch)
    with pytest.raises(RuntimeError) as caught:
        build("/device/kalfa/cuda")
    assert str(caught.value) == CUDA_MESSAGE
    with pytest.raises(RuntimeError) as caught:
        build("/device/kalfa/cuda", index=1)
    assert str(caught.value) == CUDA_MESSAGE


def test_cuda_refuses_an_index_the_machine_does_not_have(monkeypatch):
    pretend(monkeypatch, cuda=True, count=2)
    assert build("/device/kalfa/cuda", index=1) == torch.device("cuda:1")
    for index in (2, -1):
        with pytest.raises(RuntimeError) as caught:
            build("/device/kalfa/cuda", index=index)
        assert str(caught.value) == f"cuda device index {index} does not exist; this machine has 2 cuda device(s)"


def test_mps_is_the_apple_device_when_available():
    if not torch.backends.mps.is_available():
        pytest.skip("no mps device on this machine")
    device = build("/device/kalfa/mps")
    assert device == torch.device("mps") and device.type == "mps"


def test_mps_refuses_a_machine_without_it(monkeypatch):
    pretend(monkeypatch)
    with pytest.raises(RuntimeError) as caught:
        build("/device/kalfa/mps")
    assert str(caught.value) == MPS_MESSAGE


def test_auto_takes_the_first_available_of_cuda_mps_cpu(monkeypatch):
    if torch.cuda.is_available():
        expected = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        expected = torch.device("mps")
    else:
        expected = torch.device("cpu")
    assert build("/device/kalfa/auto") == expected
    pretend(monkeypatch, cuda=True, mps=True)
    assert build("/device/kalfa/auto") == torch.device("cuda:0")
    pretend(monkeypatch, mps=True)
    assert build("/device/kalfa/auto") == torch.device("mps")
    pretend(monkeypatch)
    assert build("/device/kalfa/auto") == torch.device("cpu")


def test_device_wraps_the_torch_device_with_its_lego_and_params():
    device = Device("cpu", "/device/kalfa/cpu", {"index": 0})
    assert device.torch == torch.device("cpu") and device.type == "cpu" and str(device) == "cpu"
    assert device.note() == {"device": "cpu", "uri": "/device/kalfa/cpu", "params": {"index": 0}}
    assert Device.cpu() == Device("cpu", "/device/kalfa/cpu")
    assert Device.cpu().note() == {"device": "cpu", "uri": "/device/kalfa/cpu", "params": {}}
    assert Device.cpu() != device
    assert Device.cpu() != Device("cpu", "/device/kalfa/auto")
    assert Device.cpu() != torch.device("cpu")


def test_device_moves_batches_and_places_modules():
    device = Device.cpu()
    x = torch.ones(2, 3)
    batch = {"x": x, "name": "row", "rows": [1, 2]}
    moved = device.move(batch)
    assert list(moved) == ["x", "name", "rows"]
    assert moved["x"].device.type == "cpu" and torch.equal(moved["x"], x)
    assert moved["name"] is batch["name"] and moved["rows"] is batch["rows"]
    modules = {"a": nn.Linear(3, 1), "b": nn.Linear(1, 1)}
    assert device.place(modules) is modules
    assert all(parameter.device.type == "cpu" for module in modules.values() for parameter in module.parameters())


def test_device_generator_is_seeded_from_the_global_stream():
    device = Device.cpu()
    torch.manual_seed(3)
    expected = int(torch.randint(0, 2 ** 31 - 1, (1,)))
    torch.manual_seed(3)
    generator = device.generator()
    assert generator.device.type == "cpu" and generator.initial_seed() == expected
    torch.manual_seed(3)
    again = device.generator()
    assert torch.equal(torch.rand(4, generator=generator), torch.rand(4, generator=again))
    assert device.generator().initial_seed() != expected


def test_autocast_and_the_scaler_stay_off_on_the_cpu():
    device = Device.cpu()
    assert isinstance(device.autocast(False), contextlib.nullcontext)
    assert device.scaler(False) is None
    with device.autocast(True):
        assert torch.is_autocast_enabled("cpu") and torch.get_autocast_dtype("cpu") == torch.bfloat16
        assert (torch.ones(2, 2) @ torch.ones(2, 2)).dtype == torch.bfloat16
    assert not torch.is_autocast_enabled("cpu")
    scaler = device.scaler(True)
    assert isinstance(scaler, torch.amp.GradScaler) and scaler.is_enabled() is False


def test_device_on_mps_moves_tensors_there():
    if not torch.backends.mps.is_available():
        pytest.skip("no mps device on this machine")
    device = Device("mps", "/device/kalfa/mps")
    assert device.type == "mps" and device.note()["device"] == "mps"
    assert device.move({"x": torch.ones(2)})["x"].device.type == "mps"
    assert device.generator().device.type == "mps"
    assert device.scaler(True).is_enabled() is False
