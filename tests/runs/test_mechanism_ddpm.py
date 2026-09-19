from pathlib import Path

import pytest
import torch

from helpers import config_path, copied
from kalfa.api import check, generate
from kalfa.cli import main
from kalfa.config import parse_sets
from kalfa.std.checkpoint.base import load
from kalfa.std.common.history import History


pytestmark = pytest.mark.slow


@pytest.fixture
def ddpm(trained, at_root):
    return trained("ddpm")


def test_the_document_carries_the_schedule_call_and_the_turn_extras(at_root):
    prepared = check([config_path("ddpm")], parse_sets([]))
    assert prepared.problems == [] and prepared.sizes == {"train": 64, "valid": 0, "test": 0}
    objective = prepared.document["losses"]["ddpm"]["params"]["objective"]
    assert objective["params"]["schedule"] == {"uri": "/schedule/kalfa/linear_betas", "params": {"steps": 10}}
    assert prepared.document["flow"]["training"]["params"]["turn_params"] == {"grad_clip": 1.0}


def test_the_history_is_the_train_loss_alone_and_the_schedule_ends_at_zero(ddpm):
    history = History.read(ddpm.record)
    assert [list(line) for line in history] == [["turn", "global_step", "train/ddpm", "lr/main", "minimizes/main",
                                                 "seconds", "rules"]] * 2
    assert [line["global_step"] for line in history] == [8, 16]
    assert history[0]["lr/main"] > 0.0 and history[1]["lr/main"] == 0.0
    assert not (Path(ddpm.record) / "checkpoints" / "best.pt").exists()
    final = load(Path(ddpm.record) / "final" / "state.pt")
    assert final["optimizers"]["main"]["updates"] == 16
    assert not torch.equal(final["emas"]["net"]["model.nodes.body.0.weight"],
                           final["models"]["net"]["nodes.body.0.weight"])


def test_the_samples_are_written_every_turn_and_by_the_generate_step(ddpm):
    samples = Path(ddpm.record) / "samples"
    assert sorted(path.name for path in samples.iterdir()) == ["grid.png", "samples.pt", "turn_0001.png",
                                                               "turn_0001.pt", "turn_0002.png", "turn_0002.pt"]
    generated = torch.load(samples / "samples.pt", weights_only=False)
    assert generated.shape == (4, 1, 8, 8) and float(generated.abs().max()) <= 1.0


def test_generate_runs_through_the_api_and_the_command_line(ddpm, tmp_path):
    copy = copied(ddpm.record, tmp_path / "ddpm_copy")
    assert generate(copy, device="cpu").samples.shape == (4, 1, 8, 8)
    assert main(["generate", str(copy), "--device", "cpu"]) == 0
