from pathlib import Path

import pytest
import torch

from kalfa.api import check, generate
from kalfa.cli import main
from kalfa.config import parse_sets
from kalfa.record import read_history
from runs.conftest import SMALL

pytestmark = pytest.mark.slow

SETS, PARAMS = SMALL["08_ddpm"]


def test_check_reads_the_diffusion_config(dataset):
    dataset("08_ddpm")
    prepared = check(["config.yaml"], parse_sets(SETS, PARAMS))
    assert prepared.problems == []
    assert prepared.sizes == {"train": 160, "valid": 0, "test": 0}
    document = prepared.document
    assert document["losses"]["ddpm"]["params"]["schedule"] == {"uri": "/schedule/kalfa/linear_betas",
                                                                "params": {"steps": 20}}
    assert document["flow"]["training"]["params"]["turn_params"] == {"amp": True, "grad_clip": 1.0}
    sampler = document["metrics"]["samples"]["params"]["metric"]["params"]["sampler"]
    assert sampler["uri"] == "/generate/kalfa/ddpm_sampler" and sampler["params"]["n"] == 4


def test_the_noise_objective_the_schedule_and_the_ema_samples(trained):
    result = trained("08_ddpm")
    record = Path(result.record)
    history = read_history(record)
    assert [line["turn"] for line in history] == [1, 2, 3]
    assert set(history[0]) == {"turn", "global_step", "train/ddpm", "lr/main", "rules"}
    assert [line["global_step"] for line in history] == [20, 40, 60]
    assert history[0]["lr/main"] > 0.0 and history[2]["lr/main"] == pytest.approx(0.0, abs=1e-12)
    assert history[0]["lr/main"] > history[1]["lr/main"] > history[2]["lr/main"]
    assert (record / "checkpoints" / "last.pt").exists() and not (record / "checkpoints" / "best.pt").exists()
    payload = torch.load(record / "final" / "state.pt", weights_only=False)
    assert "unet" in payload["emas"] and payload["optimizers"]["main"]["updates"] == 60
    key = next(name for name in payload["models"]["unet"] if name.endswith("last.weight"))
    assert not torch.equal(payload["emas"]["unet"]["model." + key], payload["models"]["unet"][key])
    samples = torch.load(record / "samples" / "samples.pt", weights_only=False)
    assert samples.shape == (4, 3, 32, 32) and float(samples.abs().max()) <= 1.0
    again = generate(result.record, device="cpu")
    assert again.samples.shape == (4, 3, 32, 32)
    assert main(["generate", result.record, "--device", "cpu"]) == 0
