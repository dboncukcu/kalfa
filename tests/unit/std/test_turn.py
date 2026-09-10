"""The alternating turn: weights change, only own parameters are touched, effects switch the loss."""

import functools
import math

import pytest
import torch

import kalfa  # noqa: F401
from helpers import frame, tiny_model
from kalfa.std.adapter import criterion as criterion_adapter
from kalfa.std.adapter import metric as metric_adapter
from kalfa.std.criterion import huber, mae, mse
from kalfa.std.feed import table
from kalfa.std.loader import torch as torch_loader
from kalfa.std.metric import rmse
from kalfa.std.optimizer import sgd
from kalfa.std.turn import alternating, apply_effects, effective_loss


def setup(seed=1, rows=32, lazy=False):
    model = tiny_model(seed=seed, lazy=lazy)
    optimizer = sgd({"model": model}, {"lr": 0.05}, None, "loss_mse")
    losses = {"loss_mse": criterion_adapter(mse), "loss_mae": criterion_adapter(mae),
              "loss_huber": criterion_adapter(functools.partial(huber, delta=1.0))}
    metrics = {"rmse": metric_adapter(rmse())}
    loader = torch_loader(table(frame(rows=rows)), "train", {"size": 8})
    counters = {"global_step": 0, "turn": 0}
    return model, optimizer, losses, metrics, loader, counters


def run_turn(model, optimizer, losses, metrics, loader, counters, effects=None, params=None, extra=None,
             losses_keys=None, metrics_keys=None):
    return alternating({"model": model}, {"model": optimizer}, {}, counters, {}, effects or {}, loader,
                       params or {}, extra or {}, losses, metrics, losses_keys or {}, metrics_keys or {}, "model", None,
                       device="cpu")


def test_one_turn_changes_weights_and_reports_running_means():
    model, optimizer, losses, metrics, loader, counters = setup()
    before = model.nodes["layer"].weight.clone()
    out = run_turn(model, optimizer, losses, metrics, loader, counters)
    assert not torch.equal(before, model.nodes["layer"].weight)
    assert out["models"]["model"] is model and out["counters"] is counters
    assert counters == {"global_step": 4, "turn": 1}
    assert set(out["metrics"]) == {"loss_mse", "loss_mae", "loss_huber", "rmse"}
    assert out["metrics"]["loss_mae"] > 0.0 and out["metrics"]["rmse"] == pytest.approx(out["metrics"]["loss_mse"] ** 0.5, rel=0.3)


def test_lazy_model_trains_from_the_first_batch():
    model, optimizer, losses, metrics, loader, counters = setup(lazy=True)
    run_turn(model, optimizer, losses, metrics, loader, counters)
    assert model.initialized and optimizer.real is not None


def test_effects_switch_the_active_loss():
    model, optimizer, losses, metrics, loader, counters = setup()
    assert effective_loss("model", {}, {"model": optimizer}) == "loss_mse"
    assert effective_loss("model", {"loss": "loss_mae"}, {"model": optimizer}) == "loss_mae"
    assert effective_loss("model", {"model.loss": "loss_huber", "loss": "loss_mae"}, {"model": optimizer}) == "loss_huber"
    seen = []
    original = losses["loss_mae"].loss

    def spy(context, keys=None, rescale=False):
        seen.append("mae")
        return original(context, keys)

    losses["loss_mae"].loss = spy
    run_turn(model, optimizer, losses, metrics, loader, counters, effects={"loss": "loss_mae"})
    assert len(seen) >= 4


def test_keys_select_entries_per_set_and_turn():
    model, optimizer, losses, metrics, loader, counters = setup()
    out = run_turn(model, optimizer, losses, metrics, loader, counters,
                   losses_keys={"loss_mae": {"sets": ["valid"]}, "loss_huber": {"every": 2}},
                   metrics_keys={"rmse": {"output": "y", "target": "price"}})
    assert set(out["metrics"]) == {"loss_mse", "rmse"}
    out = run_turn(model, optimizer, losses, metrics, loader, counters, losses_keys={"loss_huber": {"every": 2}})
    assert "loss_huber" in out["metrics"]


def test_apply_effects_on_params_and_trainable():
    model, optimizer, losses, metrics, loader, counters = setup()
    table = apply_effects({"loss_huber.delta": 0.2, "model.lr": 0.01, "model.trainable": False}, {"model": model},
                          {"model": optimizer}, losses)
    assert table["loss_huber"].criterion.keywords == {"delta": 0.2} and table["loss_mse"] is losses["loss_mse"]
    assert optimizer.params["lr"] == 0.01
    assert not model.kalfa_trainable and all(not parameter.requires_grad for parameter in model.parameters())
    with pytest.raises(KeyError):
        apply_effects({"ghost.x": 1}, {"model": model}, {"model": optimizer}, losses)


def test_two_optimizers_update_only_their_own_models():
    a = tiny_model(seed=1, index=0)
    b = tiny_model(seed=1, index=1)
    opt_a = sgd({"a": a}, {"lr": 0.1}, None, "l")
    opt_b = sgd({"b": b}, {"lr": 0.0}, None, "l")
    loader = torch_loader(table(frame(rows=16)), "train", {"size": 8})
    losses = {"l": criterion_adapter(mse)}
    before_b = b.nodes["layer"].weight.clone()
    alternating({"a": a, "b": b}, {"a": opt_a, "b": opt_b}, {}, {"global_step": 0, "turn": 0}, {}, {}, loader,
                {"order": ["b", "a"], "steps": {"a": 2}}, {}, losses, {}, {}, {}, "a", None)
    assert torch.equal(before_b, b.nodes["layer"].weight)
    assert b.nodes["layer"].weight.grad is None and a.nodes["layer"].weight.grad is not None


def test_turn_rejects_unknown_order_and_counts_steps_with_accumulate_and_amp():
    model, optimizer, losses, metrics, loader, counters = setup()
    with pytest.raises(KeyError):
        run_turn(model, optimizer, losses, metrics, loader, counters, params={"order": ["ghost"]})
    out = run_turn(model, optimizer, losses, metrics, loader, counters, extra={"grad_clip": 0.5, "accumulate": 2})
    assert counters["global_step"] == 2 and "loss_mse" in out["metrics"]
    before = model.nodes["layer"].weight.clone()
    out = run_turn(model, optimizer, losses, metrics, loader, counters, extra={"amp": True})
    assert counters["global_step"] == 6 and not torch.equal(before, model.nodes["layer"].weight)
    assert all(math.isfinite(value) for value in out["metrics"].values())


def test_steps_mode_keeps_the_stream_across_turns():
    model, optimizer, losses, metrics, loader, counters = setup(rows=24)
    out = alternating({"model": model}, {"model": optimizer}, {}, counters, {}, {}, loader, {}, {}, losses, metrics,
                      {}, {}, "model", {"total": 8, "turn": 2}, device="cpu")
    assert counters == {"global_step": 2, "turn": 1} and "loss_mse" in out["metrics"]
    cursor = loader.kalfa_cursor
    assert cursor is not None and not cursor.exhausted
    alternating({"model": model}, {"model": optimizer}, {}, counters, {}, {}, loader, {}, {}, losses, metrics,
                {}, {}, "model", {"total": 8, "turn": 2}, device="cpu")
    assert counters == {"global_step": 4, "turn": 2} and loader.kalfa_cursor is cursor
    for _ in range(3):
        alternating({"model": model}, {"model": optimizer}, {}, counters, {}, {}, loader, {}, {}, losses, metrics,
                    {}, {}, "model", {"total": 8, "turn": 2}, device="cpu")
    assert counters["global_step"] == 10


def test_fresh_batch_and_per_optimizer_steps_consume_batches():
    a = tiny_model(seed=1, index=0)
    b = tiny_model(seed=1, index=1)
    opt_a = sgd({"a": a}, {"lr": 0.01}, None, "l")
    opt_b = sgd({"b": b}, {"lr": 0.01}, None, "l")
    loader = torch_loader(table(frame(rows=48)), "train", {"size": 8})
    losses = {"l": criterion_adapter(mse), "lb": criterion_adapter(mse)}
    counters = {"global_step": 0, "turn": 0}
    out = alternating({"a": a, "b": b}, {"a": opt_a, "b": opt_b}, {}, counters, {}, {"b.loss": "lb"}, loader,
                      {"order": ["a", "b"], "steps": {"a": 5, "b": 1}, "fresh_batch": True}, {}, losses, {},
                      {"lb": {}}, {}, "a", None)
    assert counters["global_step"] == 1
    assert set(out["metrics"]) == {"l", "lb"}


def test_untrainable_model_stays_in_eval_mode():
    model = tiny_model(trainable=False)
    optimizer = sgd({"model": model}, {"lr": 0.1}, None, "l")
    loader = torch_loader(table(frame(rows=8)), "train", {"size": 8})
    with pytest.raises(ValueError, match="no gradient"):
        alternating({"model": model}, {"model": optimizer}, {}, {"global_step": 0, "turn": 0}, {}, {}, loader, {}, {},
                    {"l": criterion_adapter(mse)}, {}, {}, {}, "model", None)
    assert not model.training
