"""The lazy optimizer: creation on first use, own parameters only, groups and state."""

import pytest
import torch

import kalfa  # noqa: F401
from helpers import tiny_model
from kalfa.std.optimizer.torch.optimizers import Adam, AdamW, Sgd


def test_lazy_optimizer_builds_on_first_step_and_carries_loss_and_schedule():
    model = tiny_model(lazy=True)
    optimizer = Adam({"m": model}, {"lr": 0.1}, None, "mse")
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
    optimizer = Sgd({"first": first}, {"lr": 0.1}, None, "mse")
    x = torch.ones(2, 3)
    (first(x).sum() + second(x).sum()).backward()
    assert second.nodes["layer"].weight.grad is not None
    optimizer.zero_grad()
    assert first.nodes["layer"].weight.grad is None and second.nodes["layer"].weight.grad is not None


def test_groups_match_model_dot_parameter_names():
    model = tiny_model()
    optimizer = AdamW({"m": model}, {"lr": 0.1, "groups": [{"match": "m.nodes.layer.bias", "lr": 0.5}]}, None, "l")
    model(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    assert [group["lr"] for group in optimizer.param_groups] == [0.1, 0.5]
    assert dict(optimizer.named_parameters()).keys() == {"m.nodes.layer.weight", "m.nodes.layer.bias"}


def test_a_group_with_a_negative_learning_rate_climbs_its_gradient():
    from cirak.build import Graph, GraphNode

    from kalfa.std.builder.kalfa.module import Module
    from kalfa.std.layer.kalfa.multipliers import Multipliers

    model = tiny_model()
    layer = Multipliers({"a": {"epsilon": 1.0, "lmbda_init": -1.0}})
    lambdas = Module(Graph(("x",), ("lmbda",), (GraphNode("lmbda", layer, ("x",), ("lmbda",)),)), name="lambdas")
    optimizer = Adam({"m": model, "lambdas": lambdas}, {"lr": 0.1, "groups": [{"match": "lambdas.*", "lr": -0.1}]},
                     None, "l")
    x = torch.ones(1, 3)
    (model(x).sum() + 3.0 * lambdas(x).sum()).backward()
    optimizer.step()
    assert [group["lr"] for group in optimizer.param_groups] == [0.1, -0.1]
    assert "lambdas.nodes.lmbda.lmbda" in dict(optimizer.named_parameters())
    assert layer.lmbda.item() > -1.0


def with_groups():
    from cirak.build import Graph, GraphNode

    from kalfa.std.builder.kalfa.module import Module
    from kalfa.std.layer.kalfa.multipliers import Multipliers

    layer = Multipliers({"a": {"epsilon": 1.0, "lmbda_init": -1.0}})
    lambdas = Module(Graph(("x",), ("lmbda",), (GraphNode("lmbda", layer, ("x",), ("lmbda",)),)), name="lambdas")
    groups = [{"name": "lambdas", "match": "lambdas.*", "lr": -0.2},
              {"name": "bias", "match": "m.nodes.layer.bias", "weight_decay": 0.0}]
    return Adam({"m": tiny_model(), "lambdas": lambdas}, {"lr": 0.1, "weight_decay": 0.01, "groups": groups}, None,
                "l")


def group_rates(optimizer):
    return [group["lr"] for group in optimizer.param_groups]


def test_a_rule_reaches_the_default_group_a_named_group_or_every_group():
    from kalfa.std.common.history import History

    optimizer = with_groups()
    optimizer.set_param("lr", {"times": 0.5})
    assert optimizer.params["lr"] == 0.05 and optimizer.rates() == {"lambdas": -0.2, "bias": 0.05}
    x = torch.ones(1, 3)
    (optimizer.models["m"](x).sum() + optimizer.models["lambdas"](x).sum()).backward()
    optimizer.step()
    assert group_rates(optimizer) == pytest.approx([0.05, -0.2, 0.05])
    optimizer.set_param("lr", {"times": 0.5}, "*")
    assert group_rates(optimizer) == pytest.approx([0.025, -0.1, 0.025])
    optimizer.set_param("lr", 0.3, "lambdas")
    assert group_rates(optimizer) == pytest.approx([0.025, 0.3, 0.025])
    optimizer.set_param("lr", 0.4, "bi*")
    assert group_rates(optimizer) == pytest.approx([0.025, 0.3, 0.4])
    optimizer.set_param("lr", 0.2)
    assert group_rates(optimizer) == pytest.approx([0.2, 0.3, 0.4]) and optimizer.lr() == 0.2
    optimizer.set_param("weight_decay", 0.02)
    assert [group["weight_decay"] for group in optimizer.param_groups] == pytest.approx([0.02, 0.02, 0.0])
    assert optimizer.rates() == {"lambdas": 0.3, "bias": 0.4}
    line = History.line({}, {"turn": 1, "global_step": 1}, {"main": optimizer}, {})
    assert line["lr/main"] == 0.2 and line["lr/main/lambdas"] == 0.3 and line["lr/main/bias"] == 0.4
    assert History([line]).rates() == {"lr/main": [0.2], "lr/main/lambdas": [0.3], "lr/main/bias": [0.4]}
    with pytest.raises(KeyError, match="no group named"):
        optimizer.set_param("lr", 0.1, "ghost")
    with pytest.raises(KeyError, match="relative effect"):
        optimizer.set_param("momentum", {"times": 2.0})


def test_the_effects_lego_addresses_a_group_by_its_dotted_target():
    from kalfa.std.common.effects import apply_effects

    optimizer = with_groups()
    x = torch.ones(1, 3)
    (optimizer.models["m"](x).sum() + optimizer.models["lambdas"](x).sum()).backward()
    optimizer.step()
    apply_effects({"main.lr": {"times": 0.5}, "main.lambdas.lr": {"times": 0.5}}, {}, {"main": optimizer}, {})
    assert group_rates(optimizer) == pytest.approx([0.05, -0.1, 0.05])
    apply_effects({"main.*.lr": 0.01}, {}, {"main": optimizer}, {})
    assert group_rates(optimizer) == pytest.approx([0.01, 0.01, 0.01])
    with pytest.raises(KeyError, match="no group named"):
        apply_effects({"main.ghost.lr": 0.01}, {}, {"main": optimizer}, {})


def test_state_dict_round_trip_with_pending_state():
    model = tiny_model()
    optimizer = Adam({"m": model}, {"lr": 0.1}, None, "l")
    model(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    state = optimizer.state_dict()
    assert set(state) == {"torch", "updates", "base"} and state["updates"] == 1
    fresh = Adam({"m": tiny_model()}, {"lr": 0.1}, None, "l")
    fresh.load_state_dict(state)
    assert fresh.pending is state["torch"] and fresh.updates == 1
    fresh.models["m"](torch.ones(1, 3)).sum().backward()
    fresh.step()
    assert fresh.pending is None and fresh.real.state_dict()["param_groups"][0]["lr"] == 0.1 and fresh.updates == 2


def test_schedule_scales_the_learning_rate_by_updates():
    import functools

    from kalfa.std.schedule.kalfa.schedules import linear_warmup

    model = tiny_model()
    schedule = functools.partial(linear_warmup, start=0.0, end=1.0, steps=4)
    optimizer = Sgd({"m": model}, {"lr": 0.5}, schedule, "l")
    seen = []
    for _ in range(5):
        model(torch.ones(1, 3)).sum().backward()
        optimizer.step()
        seen.append(optimizer.lr())
    assert seen == pytest.approx([0.125, 0.25, 0.375, 0.5, 0.5])
    optimizer.set_param("lr", 1.0)
    assert optimizer.lr() == pytest.approx(1.0)
