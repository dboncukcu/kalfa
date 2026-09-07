"""Config 06 end to end: the vae objective with its terms in the history, rules on loss params, determinism."""

from pathlib import Path

import pytest
import torch

import kalfa  # noqa: F401
from conftest import ROOT
from kalfa.api import check, run
from kalfa.config import parse_sets
from kalfa.record import read_history
from kalfa.synthetic import write_image_folder

CONFIG = str(ROOT / "configs" / "06_vae_loss_dynamics.yaml")
FORCED = ("training.rules=[{name: half_recon, when: {uri: after_epoch, params: {at: 1}}, set: {vae_loss.w_rec: 0.5}}, "
          "{name: to_huber, when: {uri: plateau, params: {monitor: val/recon_rmse, patience: 0}}, "
          "set: {vae_loss.recon: huber}}]")


@pytest.fixture
def mnist(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_image_folder(tmp_path / "data" / "mnist", classes=("zero", "one"), per_class=24, size=28)
    return tmp_path


def test_check_resolves_the_objective_refs(mnist):
    prepared = check([CONFIG])
    assert prepared.problems == []
    losses = prepared.document["losses"]
    assert losses["vae_loss"]["params"]["recon"] == {"uri": "/criterion/kalfa/mse"}
    assert losses["vae_loss"]["params"]["kl_schedule"]["uri"] == "/schedule/kalfa/linear_warmup"
    assert losses["vae_loss"]["params"]["encoder"] == "encoder"
    rules = prepared.document["flow"]["training"]["params"]["rules"]
    assert rules[1]["set"] == {"vae_loss.recon": {"uri": "/criterion/kalfa/huber"}}
    prepared = check([CONFIG], parse_sets(["training.rules=[{name: bad, when: {uri: after_epoch, params: {at: 1}}, "
                                           "set: {vae_loss.nothing: 1}}]"]))
    assert [problem.kind for problem in prepared.problems] == ["set_value"]


def test_run_06_terms_rules_and_determinism(mnist):
    result = run([CONFIG], parse_sets(["data.batch=16", FORCED], ["epochs=3", "kl_warmup_steps=6"]), when="fixed")
    record = Path(result.record)
    history = read_history(record)
    assert [line["turn"] for line in history] == [1, 2, 3]
    keys = set(history[0])
    assert {"train/vae_loss", "train/vae_loss/recon", "train/vae_loss/kl", "train/vae_loss/w_kl", "val/vae_loss/kl",
            "val/recon_rmse", "train/recon_rmse"} <= keys
    assert 0.0 < history[0]["train/vae_loss/w_kl"] < 1.0 and history[-1]["train/vae_loss/w_kl"] == 1.0
    assert [line["rules"] for line in history] == [["half_recon", "to_huber"], [], []]
    payload = torch.load(record / "checkpoints" / "last.pt", weights_only=False)
    effects = payload["rules"]["effects"]
    assert effects["vae_loss.w_rec"] == 0.5 and callable(effects["vae_loss.recon"])
    recon = effects["vae_loss.recon"]
    assert getattr(recon, "func", recon).__name__ == "huber"
    again = run([CONFIG], parse_sets(["data.batch=16", FORCED], ["epochs=3", "kl_warmup_steps=6"]), when="again")
    assert [line["val/recon_rmse"] for line in read_history(again.record)] == [line["val/recon_rmse"] for line in history]
    assert (record / "plots" / "reconstructions.png").exists() and (record / "plots" / "loss_curve.png").exists()


def test_the_record_reloads_without_the_alias_packs(mnist):
    from kalfa.api import check, predict

    result = run([CONFIG], parse_sets([], ["epochs=1"]), when="reload")
    resolved = Path(result.record) / "resolved.yaml"
    assert "recon: /criterion/kalfa/huber" in resolved.read_text()
    assert check([str(resolved)], parse_sets([])).problems == []
    assert predict(result.record, data="data/mnist").model == "vae"
