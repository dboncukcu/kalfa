from pathlib import Path

import pytest
import torch

from kalfa.api import check, predict, run
from kalfa.config import parse_sets
from kalfa.std.common.history import History

pytestmark = pytest.mark.slow

FORCED = ("training.rules=[{name: half_recon, when: {uri: after_epoch, params: {at: 1}}, set: {vae_loss.w_rec: 0.5}}, "
          "{name: to_huber, when: {uri: plateau, params: {monitor: val/recon_rmse, patience: 0}}, "
          "set: {vae_loss.recon: huber}}]")


def test_check_resolves_the_objective_refs(dataset):
    dataset("06_vae_loss_dynamics")
    prepared = check(["config.yaml"])
    assert prepared.problems == []
    losses = prepared.document["losses"]
    vae_loss = losses["vae_loss"]["params"]["objective"]
    assert vae_loss["params"]["recon"] == {"uri": "/criterion/kalfa/mse"}
    assert vae_loss["params"]["kl_schedule"]["uri"] == "/schedule/kalfa/linear_warmup"
    assert vae_loss["params"]["encoder"] == "encoder"
    rules = prepared.document["flow"]["training"]["params"]["rules"]
    assert rules[1]["set"] == {"vae_loss.recon": {"uri": "/criterion/kalfa/huber"}}
    prepared = check(["config.yaml"], parse_sets(["training.rules=[{name: bad, when: {uri: after_epoch, "
                                                  "params: {at: 1}}, "
                                                  "set: {vae_loss.nothing: 1}}]"]))
    assert [problem.kind for problem in prepared.problems] == ["set_value"]


def test_terms_rules_and_determinism(dataset):
    dataset("06_vae_loss_dynamics")
    sets = ["data.batch=16", FORCED]
    params = ["epochs=3", "kl_warmup_steps=6"]
    result = run(["config.yaml"], parse_sets(sets, params), when="forced")
    record = Path(result.record)
    history = History.read(record)
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
    again = run(["config.yaml"], parse_sets(sets, params), when="again")
    replayed = [line["val/recon_rmse"] for line in History.read(again.record)]
    assert replayed == [line["val/recon_rmse"] for line in history]
    assert (record / "plots" / "reconstructions.png").exists() and (record / "plots" / "loss_curve.png").exists()


def test_the_record_reloads_without_the_alias_packs(trained):
    result = trained("06_vae_loss_dynamics")
    resolved = Path(result.record) / "resolved.yaml"
    assert "recon: /criterion/kalfa/huber" in resolved.read_text()
    assert check([str(resolved)], parse_sets([])).problems == []
    assert predict(result.record, data="data/mnist_new").model == "vae"
