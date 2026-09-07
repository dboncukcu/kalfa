"""Config 08 end to end: the ddpm objective with the turn's rng, amp and clipping, warmup cosine, sampling from final/."""

from pathlib import Path

import pytest
import torch

import kalfa  # noqa: F401
from conftest import ROOT
from kalfa.api import check, generate, run
from kalfa.cli import main
from kalfa.config import parse_sets
from kalfa.record import read_history
from kalfa.synthetic import write_image_folder

CONFIG = str(ROOT / "configs" / "08_ddpm.yaml")
SMALL = ["device=cpu", "data.batch=8", "optimizers.main.schedule={uri: warmup_cosine, params: {warmup: 2, total: 8}}",
         "generate.params.n=4"]
PARAMS = ["epochs=3", "diffusion_steps=20"]


@pytest.fixture
def cifar(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_image_folder(tmp_path / "data" / "cifar10", classes=("bird", "cat", "dog"), per_class=8, size=32, channels=3)
    return tmp_path


def test_check_08(cifar):
    prepared = check([CONFIG], parse_sets(SMALL, PARAMS))
    assert prepared.problems == []
    assert prepared.sizes == {"train": 24, "valid": 0, "test": 0}
    document = prepared.document
    assert document["losses"]["ddpm"]["params"]["schedule"] == {"uri": "/schedule/kalfa/linear_betas",
                                                                "params": {"steps": 20}}
    assert document["flow"]["training"]["params"]["turn_params"] == {"amp": True, "grad_clip": 1.0}
    sampler = document["metrics"]["samples"]["params"]["metric"]["params"]["sampler"]
    assert sampler["uri"] == "/generate/kalfa/ddpm_sampler" and sampler["params"]["n"] == 4


def test_run_08(cifar):
    result = run([CONFIG], parse_sets(SMALL, PARAMS), when="fixed")
    record = Path(result.record)
    history = read_history(record)
    assert [line["turn"] for line in history] == [1, 2, 3]
    assert set(history[0]) == {"turn", "global_step", "train/ddpm", "lr/main", "rules"}
    assert [line["global_step"] for line in history] == [3, 6, 9]
    assert history[0]["lr/main"] > 0.0 and history[2]["lr/main"] == pytest.approx(0.0, abs=1e-12)
    assert history[0]["lr/main"] > history[1]["lr/main"] > history[2]["lr/main"]
    assert (record / "checkpoints" / "last.pt").exists() and not (record / "checkpoints" / "best.pt").exists()
    payload = torch.load(record / "final" / "state.pt", weights_only=False)
    assert "unet" in payload["emas"] and payload["optimizers"]["main"]["updates"] == 9
    key = next(name for name in payload["models"]["unet"] if name.endswith("last.weight"))
    assert not torch.equal(payload["emas"]["unet"]["model." + key], payload["models"]["unet"][key])
    samples = torch.load(record / "samples" / "samples.pt", weights_only=False)
    assert samples.shape == (4, 3, 32, 32) and float(samples.abs().max()) <= 1.0
    again = generate(result.record, device="cpu")
    assert again.samples.shape == (4, 3, 32, 32)
    assert main(["generate", result.record, "--device", "cpu"]) == 0
