"""The lazy optimizer: creation on first use, own parameters only, groups and state."""

import pytest
import torch

import kalfa  # noqa: F401
from helpers import tiny_model
from kalfa.std.optimizer import adam, adamw, sgd


def test_lazy_optimizer_builds_on_first_step_and_carries_loss_and_schedule():
    model = tiny_model(lazy=True)
    optimizer = adam({"m": model}, {"lr": 0.1}, None, "mse")
    assert optimizer.loss == "mse" and optimizer.schedule is None and optimizer.real is None
    assert optimizer.lr() == 0.1
    x = torch.ones(2, 3)
    model(x).sum().backward()
    optimizer.step()
    assert isinstance(optimizer.real, torch.optim.Adam) and optimizer.lr() == 0.1
    optimizer.set_param("lr", 0.01)
    assert optimizer.param_groups[0]["lr"] == 0.01


def test_zero_grad_clears_own_parameters_only():
    first = tiny_model(seed=1, index=0)
    second = tiny_model(seed=1, index=1)
    optimizer = sgd({"first": first}, {"lr": 0.1}, None, "mse")
    x = torch.ones(2, 3)
    (first(x).sum() + second(x).sum()).backward()
    assert second.nodes["layer"].weight.grad is not None
    optimizer.zero_grad()
    assert first.nodes["layer"].weight.grad is None and second.nodes["layer"].weight.grad is not None


def test_groups_match_model_dot_parameter_names():
    model = tiny_model()
    optimizer = adamw({"m": model}, {"lr": 0.1, "groups": [{"match": "m.nodes.layer.bias", "lr": 0.5}]}, None, "l")
    model(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    assert [group["lr"] for group in optimizer.param_groups] == [0.1, 0.5]
    assert dict(optimizer.named_parameters()).keys() == {"m.nodes.layer.weight", "m.nodes.layer.bias"}


def test_state_dict_round_trip_with_pending_state():
    model = tiny_model()
    optimizer = adam({"m": model}, {"lr": 0.1}, None, "l")
    model(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    state = optimizer.state_dict()
    assert set(state) == {"torch", "updates", "base"} and state["updates"] == 1
    fresh = adam({"m": tiny_model()}, {"lr": 0.1}, None, "l")
    fresh.load_state_dict(state)
    assert fresh.pending is state["torch"] and fresh.updates == 1
    fresh.models["m"](torch.ones(1, 3)).sum().backward()
    fresh.step()
    assert fresh.pending is None and fresh.real.state_dict()["param_groups"][0]["lr"] == 0.1 and fresh.updates == 2


def test_schedule_scales_the_learning_rate_by_updates():
    import functools

    from kalfa.std.schedule import linear_warmup

    model = tiny_model()
    schedule = functools.partial(linear_warmup, start=0.0, end=1.0, steps=4)
    optimizer = sgd({"m": model}, {"lr": 0.5}, schedule, "l")
    seen = []
    for _ in range(5):
        model(torch.ones(1, 3)).sum().backward()
        optimizer.step()
        seen.append(optimizer.lr())
    assert seen == pytest.approx([0.125, 0.25, 0.375, 0.5, 0.5])
    optimizer.set_param("lr", 1.0)
    assert optimizer.lr() == pytest.approx(1.0)
