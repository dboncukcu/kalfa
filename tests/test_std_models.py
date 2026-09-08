"""Model legos: layers, inits, the builder and the EMA clone."""

import torch
from cirak.build import Graph, GraphNode
from torch import nn

import kalfa  # noqa: F401
from helpers import linear_graph, tiny_model
from kalfa.std.builder import Module, apply_roles, model_seed, module
from kalfa.std.init import normal, xavier, zeros
from kalfa.std.layer import concat, dropout, flatten, leaky_relu, linear, linear_relu, relu, torch_linear
from kalfa.std.model import clone


def test_layers():
    assert isinstance(linear(4), nn.LazyLinear) and isinstance(linear(4, 3), nn.Linear)
    stack = linear_relu(4)
    assert isinstance(stack[0], nn.LazyLinear) and isinstance(stack[1], nn.ReLU)
    assert torch_linear(3, 2).weight.shape == (2, 3)
    assert concat(1)(torch.ones(2, 1), torch.zeros(2, 2)).shape == (2, 3)
    assert flatten()(torch.ones(2, 3, 4)).shape == (2, 12)
    assert relu()(torch.tensor([-1.0, 1.0])).tolist() == [0.0, 1.0]
    assert isinstance(leaky_relu(0.2), nn.LeakyReLU) and isinstance(dropout(0.1), nn.Dropout)


def test_inits_apply_per_role():
    layer = nn.Linear(4, 4)
    apply_roles(layer, {"weights": zeros(), "bias": normal(std=0.5, mean=3.0)})
    assert float(layer.weight.detach().abs().sum()) == 0.0 and abs(float(layer.bias.detach().mean()) - 3.0) < 1.5
    apply_roles(layer, {}, [{"match": "weight", "weights": xavier()}])
    assert float(layer.weight.detach().abs().sum()) > 0.0


def test_builder_is_seeded_by_index_and_deterministic():
    first = tiny_model(seed=7, index=0)
    second = tiny_model(seed=7, index=0)
    third = tiny_model(seed=7, index=1)
    assert torch.equal(first.nodes["layer"].weight, second.nodes["layer"].weight)
    assert not torch.equal(first.nodes["layer"].weight, third.nodes["layer"].weight)
    assert model_seed(7, 0) != model_seed(7, 1) and model_seed(7, 0) == model_seed(7, 0)
    assert first.inputs == ["x"] and first.outputs == ["y"]
    assert first(torch.ones(2, 3)).shape == (2, 1)


def test_lazy_layers_materialize_under_the_model_seed_on_the_first_batch():
    first = tiny_model(seed=3, lazy=True)
    second = tiny_model(seed=3, lazy=True)
    assert not first.initialized
    x = torch.ones(2, 5)
    first(x)
    second(x)
    assert first.initialized and first.nodes["layer"].weight.shape == (1, 5)
    assert torch.equal(first.nodes["layer"].weight, second.nodes["layer"].weight)


def test_init_roles_and_node_init():
    graph = Graph(("x",), ("y",), (GraphNode("a", nn.Linear(3, 3), ("x",), ("h",)),
                                   GraphNode("b", nn.Linear(3, 1), ("h",), ("y",), extra={"init": {"weights": zeros()}})))
    model = Module(graph, seed=1, init={"bias": zeros()})
    assert float(model.nodes["a"].bias.detach().abs().sum()) == 0.0
    assert float(model.nodes["b"].bias.detach().abs().sum()) == 0.0
    assert float(model.nodes["b"].weight.detach().abs().sum()) == 0.0
    assert float(model.nodes["a"].weight.detach().abs().sum()) > 0.0


def test_trainable_false_freezes_and_keeps_eval_mode():
    model = tiny_model(trainable=False)
    assert all(not parameter.requires_grad for parameter in model.parameters())
    model.train()
    assert not model.training
    live = tiny_model()
    live.train()
    assert live.training and all(parameter.requires_grad for parameter in live.parameters())


def test_composite_references_models_and_owns_no_parameters():
    encoder = tiny_model(in_features=3, out_features=2, seed=1, index=0)
    graph = Graph(("x",), ("s",), (GraphNode("z", None, ("x",), ("z",), ref="encoder"),
                                   GraphNode("s", nn.Identity(), ("z",), ("s",))))
    composite = module(graph, models={"encoder": encoder})
    out = composite(torch.ones(4, 3))
    assert out.shape == (4, 2) and torch.equal(out, encoder(torch.ones(4, 3)))
    assert list(composite.parameters()) == []
    assert composite.refs["z"] is encoder


def test_load_state_dict_marks_the_model_built():
    source = tiny_model(seed=2, lazy=True)
    source(torch.ones(1, 4))
    target = tiny_model(seed=9, lazy=True)
    target.load_state_dict(source.state_dict())
    assert target.initialized
    assert torch.equal(target(torch.ones(2, 4)), source(torch.ones(2, 4)))


def test_clone_shifts_towards_the_model():
    model = tiny_model(seed=1)
    ema = clone(model, 0.5)
    with torch.no_grad():
        model.nodes["layer"].weight.fill_(1.0)
    before = ema.model.nodes["layer"].weight.clone()
    ema.shift(model)
    after = ema.model.nodes["layer"].weight
    assert torch.allclose(after, 0.5 * before + 0.5)
    assert not ema.training and all(not parameter.requires_grad for parameter in ema.parameters())
    assert ema(torch.ones(1, 3)).shape == (1, 1)
