from pathlib import Path

import pytest
import torch

from helpers import config_path, copied
from kalfa.api import check, generate
from kalfa.config import parse_sets
from kalfa.std.checkpoint.base import load
from kalfa.std.common.history import History


pytestmark = pytest.mark.slow

KEYS = ["turn", "global_step", "train/wgan_d", "train/wgan_g", "val/wgan_g", "val/fid", "lr/d", "lr/g", "minimizes/d",
        "minimizes/g", "seconds", "rules"]


@pytest.fixture
def gan(trained, at_root):
    return trained("gan")


def test_the_check_reads_the_adversarial_definitions(at_root):
    prepared = check([config_path("gan")], parse_sets([]))
    assert prepared.problems == []
    training = prepared.document["flow"]["training"]["params"]
    assert training["losses_keys"] == {"wgan_d": {"sets": ["train"]}, "wgan_g": {}}
    assert training["turn"] == {"uri": "/turn/kalfa/alternating",
                                "params": {"order": ["d", "g"], "steps": {"d": 2, "g": 1}, "fresh_batch": True}}
    assert prepared.document["flow"]["models"]["params"]["ema_items"] == [{"name": "gen", "decay": 0.9}]
    grad = check([config_path("gan")], parse_sets(["losses.wgan_d.sets=[train, valid]"]))
    assert [problem.kind for problem in grad.errors] == ["needs_grad_set"]
    missing = check([config_path("gan")], parse_sets(["generate=null"]))
    assert "generate_missing" in [problem.kind for problem in missing.errors]


def test_the_order_and_the_steps_shape_the_updates(gan):
    history = History.read(gan.record)
    assert [list(line) for line in history] == [KEYS] * 2
    assert [line["global_step"] for line in history] == [2, 4]
    assert [line["minimizes/d"] for line in history] == ["wgan_d"] * 2
    steps = History.read_steps(gan.record)
    assert len(steps) == 4 and list(steps[0]) == ["step", "turn", "loss/d", "lr/d", "loss/g", "lr/g"]
    final = load(Path(gan.record) / "final" / "state.pt")
    assert final["optimizers"]["d"]["updates"] == 8 and final["optimizers"]["g"]["updates"] == 4


def test_the_ema_copy_the_samples_and_the_plots_are_written(gan):
    best = load(Path(gan.record) / "checkpoints" / "best.pt")
    assert list(best["emas"]) == ["gen"]
    assert not torch.equal(best["emas"]["gen"]["model.nodes.s1.weight"], best["models"]["gen"]["nodes.s1.weight"])
    samples = Path(gan.record) / "samples"
    assert sorted(path.name for path in samples.iterdir()) == ["grid.png", "samples.pt", "turn_0001.png",
                                                               "turn_0001.pt", "turn_0002.png", "turn_0002.pt"]
    assert torch.load(samples / "turn_0001.pt", weights_only=False).shape == (4, 1, 8, 8)
    assert torch.load(samples / "samples.pt", weights_only=False).shape == (4, 1, 8, 8)
    plots = sorted(path.name for path in (Path(gan.record) / "plots").iterdir())
    assert plots == ["loss_curve.png", "samples_gif.gif", "samples_matrix.png"]
    assert not (Path(gan.record) / "predictions.parquet").exists()


def test_generate_samples_from_the_best_ema_generator(gan, tmp_path):
    copy = copied(gan.record, tmp_path / "gan_copy")
    generated = generate(copy, which="best")
    assert generated.path == str(copy / "samples" / "samples.pt")
    assert generated.samples.shape == (4, 1, 8, 8) and float(generated.samples.abs().max()) <= 1.0
