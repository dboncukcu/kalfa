"""Config 07 end to end on a tiny image folder: critic steps, gradient penalty, EMA, fid every five turns, samples."""

from pathlib import Path

import pytest
import torch

import kalfa  # noqa: F401
from conftest import ROOT
from kalfa.api import check, generate, run
from kalfa.config import parse_sets
from kalfa.record import read_history
from kalfa.std import metric as metric_module
from kalfa.synthetic import write_image_folder

CONFIG = str(ROOT / "configs" / "07_wgan_gp.yaml")
SMALL = ["device=cpu", "data.batch.size=4", "metrics.fid.params.n=8", "generate.params.n=4"]


def pooled_pixels(images):
    return torch.nn.functional.adaptive_avg_pool2d(images, 2).flatten(1)


@pytest.fixture
def celeba(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(metric_module, "DEFAULT_EXTRACTOR", pooled_pixels)
    write_image_folder(tmp_path / "data" / "celeba64", classes=("a", "b", "c"), per_class=16, size=32, channels=3)
    return tmp_path


def test_check_07(celeba):
    prepared = check([CONFIG], parse_sets(SMALL))
    assert prepared.problems == []
    assert prepared.sizes == {"train": 46, "valid": 2, "test": 0}
    document = prepared.document
    assert document["flow"]["training"]["params"]["losses_keys"] == {"wgan_d": {"sets": ["train"]}, "wgan_g": {}}
    assert document["flow"]["training"]["params"]["metrics_keys"] == {"fid": {"sets": ["valid"], "every": 5},
                                                                       "samples": {"sets": ["valid"], "every": 5}}
    sampler = document["metrics"]["samples"]["params"]["metric"]["params"]["sampler"]
    assert sampler["uri"] == "/generate/kalfa/gan_sampler" and sampler["params"]["n"] == 4
    prepared = check([CONFIG], parse_sets(SMALL + ["generate=null"]))
    assert "generate_missing" in [problem.kind for problem in prepared.problems]
    assert document["flow"]["models"]["params"]["ema_items"] == [{"name": "generator", "decay": 0.999}]
    assert document["flow"]["after"]["params"]["generate"]["uri"] == "/generate/kalfa/gan_sampler"
    prepared = check([CONFIG], parse_sets(SMALL + ["losses.wgan_d.sets=[train, valid]"]))
    assert "needs_grad_set" in [problem.kind for problem in prepared.problems]
    prepared = check([CONFIG], parse_sets(SMALL + ["training.amp=true"]))
    assert "amp_scaler" not in [problem.kind for problem in prepared.problems]


def test_run_07(celeba):
    result = run([CONFIG], parse_sets(SMALL, ["epochs=5"]), when="fixed")
    record = Path(result.record)
    history = read_history(record)
    assert [line["turn"] for line in history] == [1, 2, 3, 4, 5]
    assert {"train/wgan_d", "train/wgan_g", "val/wgan_g", "lr/d", "lr/g"} <= set(history[0])
    assert "val/fid" not in history[0] and "val/fid" in history[4]
    assert [line["global_step"] for line in history] == [2, 4, 6, 8, 10]
    payload = torch.load(record / "checkpoints" / "best.pt", weights_only=False)
    assert payload["checkpoint"]["best"] == pytest.approx(history[4]["val/fid"]) and payload["turn"] == 5
    assert "generator" in payload["emas"]
    last = torch.load(record / "checkpoints" / "last.pt", weights_only=False)
    ema = last["emas"]["generator"]
    live = last["models"]["generator"]
    key = next(name for name in live if name.endswith("project.weight"))
    assert not torch.equal(ema["model." + key], live[key])
    assert (record / "samples" / "samples.pt").exists() and (record / "samples" / "grid.png").exists()
    samples = torch.load(record / "samples" / "samples.pt", weights_only=False)
    assert samples.shape == (4, 3, 32, 32)
    assert not (record / "predictions.parquet").exists()
    written = generate(result.record, which="best")
    assert written.path.endswith("samples.pt") and written.samples.shape == (4, 3, 32, 32)
    assert (record / "samples" / "turn_0005.png").exists() and not (record / "samples" / "turn_0004.png").exists()
    turn = torch.load(record / "samples" / "turn_0005.pt", weights_only=False)
    assert turn.shape == (16, 3, 32, 32) and "val/samples" not in history[4]
    assert (record / "plots" / "samples_gif.gif").exists() and (record / "plots" / "samples_matrix.png").exists()
