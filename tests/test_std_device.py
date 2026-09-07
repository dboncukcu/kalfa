"""The device slot: device legos return torch devices and check availability; auto falls back to the cpu; a custom
device lego runs; the chosen device is recorded."""

import json
from pathlib import Path

import pytest
import torch

import kalfa
from helpers import minimal, write_config
from kalfa.api import KalfaError, build_device, check, device_of, predict, run
from kalfa.cli import main
from kalfa.config import parse_sets
from kalfa.std.device import auto, cpu, cuda, mps


def test_std_device_legos_check_availability(monkeypatch):
    assert cpu() == torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="cuda is not available"):
        cuda()
    with pytest.raises(RuntimeError, match="mps is not available"):
        mps()
    assert auto() == torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert cuda() == torch.device("cuda:0") and auto() == torch.device("cuda:0")
    with pytest.raises(RuntimeError, match="index 3 does not exist"):
        cuda(index=3)


def test_device_of_builds_the_config_lego(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert device_of({})[0] == torch.device("cpu")
    device, uri, params = device_of({"device": "/device/kalfa/auto"})
    assert device == torch.device("cpu") and uri == "/device/kalfa/auto" and params == {}
    with pytest.raises(KalfaError, match="cuda is not available"):
        device_of({"device": {"uri": "/device/kalfa/cuda", "params": {"index": 1}}})


def test_check_validates_the_device_call(workdir):
    problems = check([str(write_config(workdir / "a.yaml", minimal(device="nope")))], parse_sets([])).problems
    assert "unknown_alias" in [problem.kind for problem in problems]
    problems = check([str(write_config(workdir / "b.yaml", minimal(device="mse")))], parse_sets([])).problems
    assert "kind_mismatch" in [problem.kind for problem in problems]
    problems = check([str(write_config(workdir / "c.yaml", minimal(device="auto")))], parse_sets([])).problems
    assert problems == []
    prepared = check([str(write_config(workdir / "d.yaml", minimal()))], parse_sets(["device={uri: cuda, params: {index: 1}}"]))
    assert prepared.problems == []


def test_a_custom_device_lego_runs_and_the_choice_is_recorded(workdir, monkeypatch):
    @kalfa.lego("/device/test/fake", description="A test device: the cpu under another name")
    def fake(flavor="plain"):
        return torch.device("cpu")

    path = write_config(workdir / "cfg.yaml", minimal(device={"uri": "/device/test/fake", "params": {"flavor": "x"}}))
    result = run([str(path)], parse_sets([]), when="fake")
    assert result.device == torch.device("cpu")
    note = json.loads((Path(result.record) / "device.json").read_text())
    assert note == {"device": "cpu", "uri": "/device/test/fake", "params": {"flavor": "x"}}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    result = run([str(path)], parse_sets(["device=auto"]), when="auto")
    assert json.loads((Path(result.record) / "device.json").read_text())["uri"] == "/device/kalfa/auto"
    with pytest.raises(KalfaError, match="cuda is not available"):
        run([str(path)], parse_sets(["device=cuda"]), when="cuda")


def test_predict_takes_a_device(workdir, monkeypatch):
    path = write_config(workdir / "cfg.yaml", minimal(training__epochs=1))
    result = run([str(path)], parse_sets([]), when="dev")
    plain = predict(result.record)
    on_cpu = predict(result.record, device="cpu")
    assert len(on_cpu.table) == len(plain.table)
    assert on_cpu.table["pred_y"].tolist() == plain.table["pred_y"].tolist()
    assert predict(result.record, device={"uri": "/device/kalfa/cpu"}).model == plain.model
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(KalfaError, match="cuda is not available"):
        predict(result.record, device="cuda")
    with pytest.raises(KalfaError, match="not a known device lego"):
        predict(result.record, device="nope")
    assert main(["predict", result.record, "--device", "cpu"]) == 0
    assert main(["predict", result.record, "--device", "{uri: cpu}"]) == 0
    assert main(["predict", result.record, "--device", "cuda"]) == 1


def test_the_device_flag_reads_yaml():
    from argparse import Namespace

    from kalfa.cli import _device_value

    assert _device_value(Namespace(device=None)) is None
    assert _device_value(Namespace(device="cuda")) == "cuda"
    assert _device_value(Namespace(device="{uri: cuda, params: {index: 1}}")) == {"uri": "cuda",
                                                                                  "params": {"index": 1}}
    assert build_device("cpu")[1] == "/device/kalfa/cpu"
