import math

import pytest
import torch
from torch import nn

from kalfa.core import aliases
from kalfa.std.init import normal_init, xavier_init
from kalfa.std.models import LastStep, Mlp, Sequential


def linears(module):
    return [layer for layer in module.modules() if isinstance(layer, nn.Linear)]


def registered(name):
    table = aliases.ALIASES
    return name in table or any(getattr(entry, "name", None) == name for entry in table.values())


def test_mlp_infers_its_input_width():
    model = Mlp([16, 8], out=1)
    out = model(torch.randn(5, 7))
    assert out.shape == (5, 1)
    assert linears(model)[0].in_features == 7
    assert [layer.out_features for layer in linears(model)] == [16, 8, 1]


def test_mlp_with_in_features_activation_and_dropout():
    model = Mlp([16], 3, activation="gelu", dropout=0.1, in_features=7)
    assert [type(layer) for layer in model.layers] == [nn.Linear, nn.GELU, nn.Dropout, nn.Linear]
    assert model(torch.randn(4, 7)).shape == (4, 3)


def test_mlp_accepts_a_single_hidden_size_and_a_module_activation():
    model = Mlp(8, 2, activation=nn.Tanh(), in_features=3)
    assert isinstance(model.layers[1], nn.Tanh)
    assert model(torch.randn(2, 3)).shape == (2, 2)


def test_mlp_rejects_an_unknown_activation():
    with pytest.raises(ValueError, match="unknown activation"):
        Mlp([4], 1, activation="swoosh", in_features=2)


def test_sequential_resolves_short_names_and_takes_the_last_gru_step(monkeypatch):
    table = {"relu": nn.ReLU, "flatten": nn.Flatten, "last_step": "kalfa.std.models.LastStep"}
    monkeypatch.setattr(aliases, "resolve", lambda name: table[name])
    monkeypatch.setattr(aliases, "import_target", lambda path: {"kalfa.std.models.LastStep": LastStep}[path])
    model = Sequential([nn.GRU(4, 6, batch_first=True), "last_step", nn.Linear(6, 2), "relu"])
    assert model(torch.randn(3, 10, 4)).shape == (3, 2)
    assert isinstance(model.layers[1], LastStep)
    assert isinstance(model.layers[3], nn.ReLU)
    flat = Sequential(["flatten", nn.Linear(12, 3)])
    assert flat(torch.randn(2, 3, 2, 2)).shape == (2, 3)


def test_sequential_rejects_items_that_are_not_layers():
    with pytest.raises(TypeError):
        Sequential([3.5])
    with pytest.raises(TypeError):
        Sequential("relu")


def test_last_step_reads_a_tuple_or_a_tensor():
    value = torch.arange(24.0).reshape(2, 3, 4)
    assert torch.equal(LastStep()(value), value[:, -1, :])
    assert torch.equal(LastStep()((value, None)), value[:, -1, :])


def test_normal_init_fills_weights_and_zeroes_biases():
    model = Mlp([64], 2, in_features=32)
    before = model.layers[0].weight.detach().clone()
    normal_init(std=0.02)(model)
    weight = model.layers[0].weight.detach()
    assert not torch.equal(weight, before)
    assert abs(float(weight.std()) - 0.02) < 0.005
    assert all(float(layer.bias.detach().abs().max()) == 0.0 for layer in linears(model))


def test_normal_init_fills_a_lazy_layer_on_its_first_forward():
    model = Mlp([64], 2)
    normal_init(std=0.02)(model)
    model(torch.randn(5, 32))
    first = linears(model)[0]
    assert abs(float(first.weight.detach().std()) - 0.02) < 0.005
    assert float(first.bias.detach().abs().max()) == 0.0
    with torch.no_grad():
        first.weight.zero_()
    model(torch.randn(5, 32))
    assert float(first.weight.detach().abs().max()) == 0.0


def test_xavier_init_covers_embedding_and_convolution():
    model = nn.Sequential(nn.Embedding(10, 4), nn.Conv2d(1, 2, 3), nn.Linear(3, 3))
    before = [parameter.detach().clone() for parameter in model.parameters()]
    xavier_init()(model)
    for weight, old in zip([model[0].weight, model[1].weight, model[2].weight], [before[0], before[1], before[3]]):
        assert not torch.equal(weight.detach(), old)
    assert float(model[1].bias.detach().abs().max()) == 0.0
    assert float(model[2].bias.detach().abs().max()) == 0.0
    assert float(model[0].weight.detach().abs().max()) <= math.sqrt(6.0 / (10 + 4))


def test_short_names_are_registered():
    for name in ("mlp", "stack", "linear", "relu", "gelu", "tanh", "sigmoid", "silu", "dropout", "flatten",
                 "layer_norm", "embedding", "batch_norm", "conv2d", "last_step", "normal_init", "xavier_init"):
        assert registered(name), name
