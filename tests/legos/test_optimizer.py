import math

import pytest
import torch
from cirak.build import Graph, GraphNode
from cirak.registry import registry
from torch import nn

from helpers import build, tiny_model
from kalfa.std import STD_URIS
from kalfa.std.builder.kalfa.module import Module
from kalfa.std.optimizer.base import Optimizer
from kalfa.std.optimizer.torch.optimizers import Adam, AdamW, Sgd


OPTIMIZERS = sorted(uri for uri in STD_URIS if uri.startswith("/optimizer/"))


def sgd(model, **params):
    return build("/optimizer/torch/sgd", models={"m": model}, params=params)


def grouped(model, **extra):
    params = {"lr": 0.1, "groups": [{"name": "bias", "match": "*.bias", "lr": -0.05}, {"name": "plain",
                                                                                          "match": "*.weight"}]}
    params.update(extra)
    return sgd(model, **params)


def layer(model):
    return model.nodes["layer"]


def test_optimizer_scope_is_the_three_torch_optimizers():
    assert OPTIMIZERS == ["/optimizer/torch/adam", "/optimizer/torch/adamw", "/optimizer/torch/sgd"]


def test_every_optimizer_carries_state_refs_and_its_alias():
    for uri in OPTIMIZERS:
        facts = registry.facts(uri)
        assert facts.state is True
        assert facts.refs == {"loss": "loss", "schedule": "schedule"}
        assert facts.aliases == ("models",)
        assert registry.aliases()[uri.rsplit("/", 1)[1]] == uri


def test_optimizers_wrap_their_torch_class_under_its_lower_name():
    model = tiny_model()
    assert isinstance(sgd(model, lr=0.1), Sgd) and isinstance(sgd(model, lr=0.1), Optimizer)
    assert isinstance(build("/optimizer/torch/adam", models={"m": model}, params={"lr": 0.1}), Adam)
    assert isinstance(build("/optimizer/torch/adamw", models={"m": model}, params={"lr": 0.1}), AdamW)
    assert sgd(model, lr=0.1).name == "sgd"
    assert build("/optimizer/torch/adam", models={"m": model}).name == "adam"
    assert build("/optimizer/torch/adamw", models={"m": model}).name == "adamw"


def test_optimizer_keeps_its_loss_and_schedule_names():
    schedule = build("/schedule/kalfa/step_decay", step_size=2, gamma=0.5)
    optimizer = build("/optimizer/torch/sgd", models={"m": tiny_model()}, params={"lr": 0.1}, loss="mse",
                      schedule=schedule)
    assert optimizer.loss == "mse"
    assert optimizer.schedule is schedule
    assert list(optimizer.models) == ["m"]


def test_torch_optimizer_is_created_lazily_and_once():
    optimizer = sgd(tiny_model(), lr=0.1)
    assert optimizer.real is None
    assert optimizer.lr() == 0.1
    assert optimizer.updates == 0
    real = optimizer.torch()
    assert isinstance(real, torch.optim.SGD)
    assert optimizer.torch() is real
    assert optimizer.param_groups is real.param_groups
    assert optimizer.base == [0.1]
    assert optimizer.placed == [0]
    assert math.isnan(sgd(tiny_model()).lr())


def test_parameters_are_named_after_the_model_and_its_nodes():
    optimizer = sgd(tiny_model(), lr=0.1)
    assert [name for name, _ in optimizer.named_parameters()] == ["m.nodes.layer.weight", "m.nodes.layer.bias"]
    assert len(optimizer.parameters()) == 2


def test_named_groups_split_the_parameters_by_their_match():
    model = tiny_model()
    optimizer = grouped(model)
    entries, defaults = optimizer.groups()
    assert defaults == {"lr": 0.1}
    assert entries[0] == {"params": []}
    assert entries[1]["params"] == [layer(model).bias] and entries[1]["lr"] == -0.05
    assert entries[2]["params"] == [layer(model).weight] and "lr" not in entries[2]
    assert optimizer.rates() == {"bias": -0.05, "plain": 0.1}
    optimizer.torch()
    assert optimizer.placed == [None, 0, 1]
    assert [group["lr"] for group in optimizer.param_groups] == [-0.05, 0.1]
    assert optimizer.rates() == {"bias": -0.05, "plain": 0.1}


def test_a_negative_group_lr_climbs_the_gradient():
    model = tiny_model()
    optimizer = grouped(model)
    with torch.no_grad():
        layer(model).weight.fill_(1.0)
        layer(model).bias.fill_(1.0)
    optimizer.torch()
    layer(model).weight.grad = torch.ones(1, 3)
    layer(model).bias.grad = torch.ones(1)
    optimizer.step()
    assert optimizer.updates == 1
    assert torch.allclose(layer(model).weight, torch.full((1, 3), 0.9))
    assert torch.allclose(layer(model).bias, torch.tensor([1.05]))


def test_a_group_that_matches_nothing_is_left_out_of_the_torch_optimizer():
    optimizer = sgd(tiny_model(), lr=0.1, groups=[{"name": "none", "match": "zz.*", "lr": 0.5}])
    assert optimizer.rates() == {"none": 0.5}
    optimizer.torch()
    assert len(optimizer.param_groups) == 1
    assert optimizer.placed == [0, None]
    assert optimizer.rates() == {}


def test_a_group_without_a_name_has_no_rate():
    optimizer = sgd(tiny_model(), lr=0.1, groups=[{"match": "*.bias", "lr": 0.5}])
    assert optimizer.rates() == {}
    optimizer.torch()
    assert optimizer.rates() == {}
    assert [group["lr"] for group in optimizer.param_groups] == [0.1, 0.5]


def test_an_optimizer_without_parameters_cannot_be_created():
    graph = Graph(("x",), ("y",), (GraphNode("act", nn.ReLU(), ("x",), ("y",)),))
    optimizer = sgd(Module(graph, seed=1, name="empty"), lr=0.1)
    with pytest.raises(ValueError, match=r"the optimizer has no parameters; its models have none yet"):
        optimizer.torch()


def test_set_param_on_the_default_reaches_the_groups_without_their_own_value():
    optimizer = grouped(tiny_model())
    optimizer.set_param("lr", 0.2)
    assert optimizer.params["lr"] == 0.2
    assert optimizer.rates() == {"bias": -0.05, "plain": 0.2}
    optimizer.torch()
    optimizer.set_param("lr", 0.4)
    assert [group["lr"] for group in optimizer.param_groups] == [-0.05, 0.4]
    assert optimizer.base == [-0.05, 0.4]
    assert optimizer.lr() == -0.05
    assert optimizer.rates() == {"bias": -0.05, "plain": 0.4}


def test_set_param_relative_scales_or_shifts_the_current_value():
    optimizer = grouped(tiny_model())
    optimizer.set_param("lr", {"times": 0.5})
    assert optimizer.params["lr"] == 0.05
    optimizer.set_param("lr", {"plus": 0.15})
    assert optimizer.params["lr"] == 0.2
    optimizer.set_param("lr", {"times": 2.0}, "bias")
    assert optimizer.group_specs()[0]["lr"] == -0.1
    assert optimizer.rates() == {"bias": -0.1, "plain": 0.2}
    optimizer.torch()
    optimizer.set_param("lr", {"times": 0.5}, "bias")
    assert [group["lr"] for group in optimizer.param_groups] == [-0.05, 0.2]


def test_set_param_on_every_group_skips_the_ones_that_inherit():
    optimizer = grouped(tiny_model())
    optimizer.set_param("lr", {"times": 2.0}, "*")
    assert optimizer.params["lr"] == 0.2
    assert optimizer.group_specs()[0]["lr"] == -0.1
    assert "lr" not in optimizer.group_specs()[1]
    assert optimizer.rates() == {"bias": -0.1, "plain": 0.2}
    optimizer.set_param("lr", 1.0, "pl*")
    assert optimizer.group_specs()[1]["lr"] == 1.0
    assert optimizer.rates() == {"bias": -0.1, "plain": 1.0}


def test_set_param_writes_other_hyperparameters_to_the_torch_groups():
    optimizer = sgd(tiny_model(), lr=0.1, momentum=0.5)
    optimizer.torch()
    optimizer.set_param("momentum", 0.9)
    assert optimizer.param_groups[0]["momentum"] == 0.9
    assert optimizer.params["momentum"] == 0.9


def test_set_param_names_an_unknown_group():
    optimizer = grouped(tiny_model())
    with pytest.raises(KeyError, match=r"the optimizer has no group named 'zz'; the named groups are "
                                        r"\['bias', 'plain'\]"):
        optimizer.set_param("lr", 0.1, "zz")


def test_set_param_refuses_a_relative_effect_that_is_not_times_or_plus():
    optimizer = grouped(tiny_model())
    with pytest.raises(KeyError, match=r"a relative effect is \{times: x\} or \{plus: x\} with a number, got "
                                        r"\['scale'\]"):
        optimizer.set_param("lr", {"scale": 2.0})


def test_set_param_refuses_a_relative_effect_without_a_value_to_change():
    optimizer = grouped(tiny_model())
    with pytest.raises(KeyError, match=r"a relative effect on 'momentum' needs a value to change, and the optimizer "
                                        r"has no 'momentum'"):
        optimizer.set_param("momentum", {"times": 2.0})


def test_schedule_scales_the_base_rate_by_the_update_count():
    schedule = build("/schedule/kalfa/step_decay", step_size=2, gamma=0.5)
    model = tiny_model()
    optimizer = build("/optimizer/torch/sgd", models={"m": model}, params=grouped(model).params, schedule=schedule)
    optimizer.torch()
    assert [group["lr"] for group in optimizer.param_groups] == [-0.05, 0.1]
    layer(model).weight.grad = torch.zeros(1, 3)
    layer(model).bias.grad = torch.zeros(1)
    optimizer.step()
    optimizer.step()
    assert optimizer.updates == 2
    assert [group["lr"] for group in optimizer.param_groups] == [-0.025, 0.05]
    assert optimizer.base == [-0.05, 0.1]
    assert optimizer.lr() == -0.025
    assert optimizer.rates() == {"bias": -0.025, "plain": 0.05}
    optimizer.set_param("lr", 0.2)
    assert optimizer.base == [-0.05, 0.2]
    assert [group["lr"] for group in optimizer.param_groups] == [-0.025, 0.1]


def test_zero_grad_drops_every_gradient():
    model = tiny_model()
    optimizer = sgd(model, lr=0.1)
    layer(model).weight.grad = torch.ones(1, 3)
    optimizer.zero_grad()
    assert layer(model).weight.grad is None and layer(model).bias.grad is None


def test_state_dict_before_creation_holds_no_torch_state():
    optimizer = sgd(tiny_model(), lr=0.1)
    assert optimizer.state_dict() == {"torch": None, "updates": 0, "base": None}
    optimizer.load_state_dict(None)
    assert optimizer.pending is None


def test_state_dict_round_trips_through_a_pending_load_before_creation():
    first = tiny_model()
    source = sgd(first, lr=0.1, momentum=0.9)
    source.torch()
    layer(first).weight.grad = torch.ones(1, 3)
    layer(first).bias.grad = torch.ones(1)
    source.step()
    state = source.state_dict()
    assert state["updates"] == 1 and state["base"] == [0.1]
    assert state["torch"] == source.real.state_dict()
    target = sgd(tiny_model(), lr=0.1, momentum=0.9)
    target.load_state_dict(state)
    assert target.updates == 1 and target.base == [0.1] and target.pending is state["torch"]
    assert target.real is None
    target.torch()
    assert target.pending is None
    buffers = target.real.state_dict()["state"]
    assert torch.equal(buffers[0]["momentum_buffer"], torch.ones(1, 3))
    assert torch.equal(buffers[1]["momentum_buffer"], torch.ones(1))


def test_state_dict_loads_into_a_created_optimizer_and_reschedules():
    schedule = build("/schedule/kalfa/step_decay", step_size=2, gamma=0.5)
    optimizer = build("/optimizer/torch/sgd", models={"m": tiny_model()}, params={"lr": 0.1}, schedule=schedule)
    optimizer.torch()
    assert optimizer.lr() == 0.1
    optimizer.load_state_dict({"torch": optimizer.real.state_dict(), "updates": 2, "base": [0.1]})
    assert optimizer.updates == 2
    assert optimizer.lr() == 0.05
    optimizer.load_state_dict(optimizer.real.state_dict())
    assert optimizer.updates == 2 and optimizer.lr() == 0.05


def test_a_pending_state_is_reported_until_the_optimizer_exists():
    source = sgd(tiny_model(), lr=0.1)
    source.torch()
    inner = source.real.state_dict()
    target = sgd(tiny_model(), lr=0.1)
    target.load_state_dict({"torch": inner, "updates": 3, "base": [0.1]})
    assert target.state_dict() == {"torch": inner, "updates": 3, "base": [0.1]}
