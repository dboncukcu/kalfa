import math

import pytest
import torch
from cirak.registry import registry
from torch import nn

from helpers import build
from kalfa.std import STD_URIS


INIT_URIS = sorted(uri for uri in STD_URIS if uri.startswith("/init/"))


def test_the_init_pack_holds_the_five_torch_initializers():
    assert INIT_URIS == ["/init/torch/kaiming", "/init/torch/kaiming_uniform", "/init/torch/normal",
                         "/init/torch/xavier", "/init/torch/zeros"]


@pytest.mark.parametrize("uri", INIT_URIS, ids=[uri.rsplit("/", 1)[1] for uri in INIT_URIS])
def test_init_alias_is_its_name(uri):
    assert registry.aliases()[uri.rsplit("/", 1)[1]] == uri
    assert registry.facts(uri).partial is False


def test_zeros_writes_zeros_in_place():
    apply = build("/init/torch/zeros")
    tensor = torch.ones(3, 4)
    assert apply(tensor) is None
    assert tensor.tolist() == [[0.0] * 4] * 3
    parameter = nn.Linear(4, 2).weight
    with torch.no_grad():
        apply(parameter)
    assert parameter.tolist() == [[0.0] * 4] * 2 and parameter.requires_grad


def test_normal_draws_around_mean_with_std():
    apply = build("/init/torch/normal", std=0.5, mean=3.0)
    tensor = torch.empty(400, 500)
    torch.manual_seed(0)
    apply(tensor)
    assert float(tensor.mean()) == pytest.approx(3.0, abs=0.01)
    assert float(tensor.std()) == pytest.approx(0.5, abs=0.01)
    assert tensor.unique().numel() > 100000
    again = torch.empty(400, 500)
    torch.manual_seed(0)
    apply(again)
    assert torch.equal(again, tensor)
    centred = torch.empty(400, 500)
    build("/init/torch/normal", std=0.1)(centred)
    assert float(centred.mean()) == pytest.approx(0.0, abs=0.005)
    assert float(centred.std()) == pytest.approx(0.1, abs=0.002)


def test_xavier_is_uniform_within_the_fan_bound_scaled_by_gain():
    tensor = torch.empty(300, 500)
    torch.manual_seed(1)
    build("/init/torch/xavier")(tensor)
    bound = math.sqrt(6.0 / (500 + 300))
    assert float(tensor.abs().max()) <= bound
    assert float(tensor.abs().max()) > 0.99 * bound
    assert float(tensor.std()) == pytest.approx(bound / math.sqrt(3.0), abs=0.001)
    doubled = torch.empty(300, 500)
    torch.manual_seed(1)
    build("/init/torch/xavier", gain=2.0)(doubled)
    assert torch.allclose(doubled, 2.0 * tensor)


def test_kaiming_normal_scales_its_std_by_the_nonlinearity_gain():
    tensor = torch.empty(2000, 100)
    torch.manual_seed(2)
    build("/init/torch/kaiming")(tensor)
    assert float(tensor.std()) == pytest.approx(math.sqrt(2.0 / 100), abs=0.002)
    assert float(tensor.mean()) == pytest.approx(0.0, abs=0.002)
    plain = torch.empty(2000, 100)
    torch.manual_seed(2)
    build("/init/torch/kaiming", nonlinearity="linear")(plain)
    assert float(plain.std()) == pytest.approx(0.1, abs=0.002)
    assert torch.allclose(plain * math.sqrt(2.0), tensor)
    hyperbolic = torch.empty(2000, 100)
    build("/init/torch/kaiming", nonlinearity="tanh")(hyperbolic)
    assert float(hyperbolic.std()) == pytest.approx(5.0 / 3.0 / 10.0, abs=0.003)


def test_kaiming_uniform_stays_within_the_gain_bound():
    tensor = torch.empty(2000, 100)
    torch.manual_seed(3)
    build("/init/torch/kaiming_uniform")(tensor)
    bound = math.sqrt(2.0) * math.sqrt(3.0 / 100)
    assert float(tensor.abs().max()) <= bound
    assert float(tensor.abs().max()) > 0.999 * bound
    assert float(tensor.std()) == pytest.approx(bound / math.sqrt(3.0), abs=0.002)
    plain = torch.empty(2000, 100)
    torch.manual_seed(3)
    build("/init/torch/kaiming_uniform", nonlinearity="linear")(plain)
    assert float(plain.abs().max()) <= math.sqrt(3.0 / 100)
    assert torch.allclose(plain * math.sqrt(2.0), tensor)
