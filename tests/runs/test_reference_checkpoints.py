from pathlib import Path

import pytest
import torch

from kalfa.std.checkpoint.base import load
from kalfa.std.common.history import History


pytestmark = pytest.mark.slow

MODELS = ["tower", "head_lin", "head_aux", "tail_stem", "tail_head", "lambdas"]
TOWER = ["nodes.norm.weight", "nodes.norm.bias", "nodes.stem.weight", "nodes.stem.bias"] + [
    key for block in ("h_0", "h_1") for key in (f"nodes.{block}__f.weight", f"nodes.{block}__f.bias",
                                                f"nodes.{block}__n.weight", f"nodes.{block}__n.bias",
                                                f"nodes.{block}__n.running_mean", f"nodes.{block}__n.running_var",
                                                f"nodes.{block}__n.num_batches_tracked")]


def payloads(record):
    return {name: load(Path(record) / path) for name, path in (("best", "checkpoints/best.pt"),
                                                                 ("last", "checkpoints/last.pt"),
                                                                 ("final", "final/state.pt"))}


def test_every_payload_carries_the_same_sections(reference):
    for payload in payloads(reference.record).values():
        assert list(payload) == ["models", "optimizers", "emas", "counters", "rules", "checkpoint", "rng", "turn"]
        assert list(payload["models"]) == MODELS
        assert list(payload["emas"]) == ["tower"]
        assert set(payload["rng"]) >= {"python", "numpy", "torch"}


def test_the_best_checkpoint_is_the_turn_with_the_lowest_monitor(reference):
    history = History.read(reference.record)
    best = payloads(reference.record)["best"]
    values = [line["val/rmse_lin"] for line in history]
    assert best["checkpoint"] == {"best": min(values)}
    assert best["turn"] == values.index(min(values)) + 1
    assert best["counters"] == {"global_step": history[best["turn"] - 1]["global_step"], "turn": best["turn"]}


def test_last_and_final_hold_the_end_of_training(reference):
    history = History.read(reference.record)
    found = payloads(reference.record)
    assert found["last"]["turn"] == 3 and found["final"]["turn"] == 3
    assert found["last"]["counters"] == {"global_step": history[-1]["global_step"], "turn": 3}
    assert found["final"]["checkpoint"] is None
    for name in MODELS:
        for key, value in found["last"]["models"][name].items():
            assert torch.equal(value, found["final"]["models"][name][key])


def test_the_rules_state_remembers_what_fired_and_what_is_in_force(reference):
    rules = payloads(reference.record)["final"]["rules"]
    assert rules["sticky"] == ["unfreeze", "swap_aux", "reweight"]
    assert rules["ever"] == ["unfreeze", "swap_aux", "cool_stem", "reweight"]
    assert rules["fired"] == ["cool_stem", "reweight"]
    assert rules["effects"] == {"tail_stem.trainable": True, "aux.loss": "bce_pos", "main.stem.lr": {"times": 0.5},
                                "ws.terms.mse_lin": 2.0, "total.constraints.mae_lin.scale": 2.0}
    assert rules["stop_fired"] == [] and list(rules["stop"][0]) == ["started"]


def test_the_optimizer_states_count_their_updates_and_keep_the_group_rates(reference):
    history = History.read(reference.record)
    optimizers = payloads(reference.record)["final"]["optimizers"]
    assert optimizers["main"]["updates"] == history[-1]["global_step"]
    assert optimizers["aux"]["updates"] == 2 * history[-1]["global_step"]
    assert optimizers["main"]["base"] == [1e-3, -1e-2, 2.5e-4] and optimizers["aux"]["base"] == [0.05]
    assert len(optimizers["main"]["torch"]["param_groups"]) == 3


def test_the_model_states_reflect_the_graph_the_repeat_and_the_data_bound_layers(reference):
    models = payloads(reference.record)["final"]["models"]
    assert list(models["tower"]) == TOWER
    assert models["tower"]["nodes.norm.weight"].shape == (18,)
    assert list(models["head_lin"]) == ["nodes.base.index", "nodes.delta.0.weight", "nodes.delta.0.bias",
                                        "nodes.delta.3.weight", "nodes.delta.3.bias"]
    assert models["head_lin"]["nodes.base.index"].tolist() == [0, 1]
    assert list(models["lambdas"]) == ["nodes.s0.lmbda"]
    assert models["lambdas"]["nodes.s0.lmbda"].shape == (2,)
    assert not torch.equal(models["lambdas"]["nodes.s0.lmbda"], torch.zeros(2))
    assert list(models["tail_head"]) == ["nodes.s0.weight", "nodes.s0.bias"]


def test_the_ema_copy_tracks_the_tower_without_being_it(reference):
    final = payloads(reference.record)["final"]
    ema = final["emas"]["tower"]
    assert list(ema) == [f"model.{key}" for key in TOWER]
    assert not torch.equal(ema["model.nodes.stem.weight"], final["models"]["tower"]["nodes.stem.weight"])
    assert ema["model.nodes.stem.weight"].shape == final["models"]["tower"]["nodes.stem.weight"].shape
