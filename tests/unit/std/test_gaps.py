"""The small gaps: normalisation layers, kaiming uniform, weighted mse, feature width, the feature index, the
arithmetic wires, select, the multipliers, logit, the median std scaler, the whole set as one batch, drop_last auto,
the seconds of a turn, the git note and the final weights."""

import json

import numpy
import pytest
import torch
from torch import nn

import kalfa  # noqa: F401
from helpers import frame, tiny_model
from kalfa.api import git_note, weights_of, write_git_note
from kalfa.std.common.history import History
from kalfa.std.criterion.kalfa.regression import weighted_mse
from kalfa.std.data.kalfa.components import feature_index, feature_width, target_weights
from kalfa.std.feed.kalfa.table import table
from kalfa.std.init.torch.initializers import kaiming_uniform
from kalfa.std.layer.kalfa.features import select
from kalfa.std.layer.kalfa.multipliers import Multipliers
from kalfa.std.layer.kalfa.wires import Add, Divide, Multiply, Negate, Subtract
from kalfa.std.layer.torch.normalization import batch_norm, group_norm, layer_norm
from kalfa.std.loader.kalfa.torch import torch_loader
from kalfa.std.optimizer.torch.optimizers import Sgd
from kalfa.std.pre.kalfa.median_std_scaler import MedianStdScaler
from kalfa.std.pre.kalfa.scales import Logit


def test_the_normalisation_layers_build_and_run():
    x = torch.randn(4, 6)
    assert batch_norm()(x).shape == (4, 6) and batch_norm(dims=2)(torch.randn(2, 3, 4, 4)).shape == (2, 3, 4, 4)
    assert layer_norm(6)(x).shape == (4, 6) and group_norm(2, 6)(torch.randn(2, 6, 3)).shape == (2, 6, 3)
    with pytest.raises(ValueError):
        batch_norm(dims=4)
    weight = torch.empty(8, 4)
    kaiming_uniform()(weight)
    assert float(weight.abs().max()) <= (6.0 / 4) ** 0.5 + 1e-6 and isinstance(layer_norm(6), nn.LayerNorm)


def test_weighted_mse_and_the_target_weights_follow_the_target_columns():
    loader = torch_loader(table(frame(rows=8)), "train", 4)
    assert target_weights(loader, {"price": 3.0}).tolist() == [3.0]
    assert target_weights(loader, {"pr*": 2.0, "default": 0.5}, target="price").tolist() == [2.0]
    assert target_weights(loader, {"default": 0.5}).tolist() == [0.5]
    with pytest.raises(ValueError):
        target_weights(loader, {}, target="ghost")
    predictions = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    targets = torch.zeros(2, 2)
    assert float(weighted_mse(predictions, targets, [1.0, 0.0])) == pytest.approx(2.5)
    with pytest.raises(ValueError):
        weighted_mse(predictions, targets, [1.0])
    assert feature_width(loader) == 3


def test_the_arithmetic_wires_combine_and_broadcast():
    first = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    second = torch.tensor([[10.0, 10.0], [20.0, 20.0]])
    column = torch.tensor([[2.0], [4.0]])
    assert Add()(first, second, column).tolist() == [[13.0, 14.0], [27.0, 28.0]]
    assert Add()(first).tolist() == first.tolist()
    assert Subtract()(second, first).tolist() == [[9.0, 8.0], [17.0, 16.0]]
    assert Multiply()(first, column).tolist() == [[2.0, 4.0], [12.0, 16.0]]
    assert Divide()(second, column).tolist() == [[5.0, 5.0], [5.0, 5.0]]
    assert Negate()(first).tolist() == [[-1.0, -2.0], [-3.0, -4.0]]
    assert float(Divide()(first, torch.zeros(2, 2))[0, 0]) == float("inf")


def test_select_takes_the_named_columns_in_the_order_written():
    loader = torch_loader(table(frame(rows=8)), "train", 4)
    assert feature_index(loader, "x?") == [0, 1, 2]
    assert feature_index(loader, ["x2", "x0"]) == [2, 0]
    assert feature_index(loader, ["x1"]) == [1]
    with pytest.raises(ValueError):
        feature_index(loader, "ghost*")
    values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert select([2, 0])(values).tolist() == [[3.0, 1.0], [6.0, 4.0]]
    assert select([0], dim=0)(values).tolist() == [[1.0, 2.0, 3.0]]
    assert "index" in dict(select([1]).named_buffers())


def test_the_multipliers_start_where_their_constraints_say_and_ignore_the_input():
    constraints = {"a_mean": {"epsilon": 1.0, "lmbda_init": -1.0}, "b.all": {"epsilon": 0.0}}
    layer = Multipliers(constraints, init=0.5)
    assert layer.names == ["a_mean", "b.all"] and layer.lmbda.tolist() == [-1.0, 0.5]
    assert layer(torch.zeros(4, 3)).tolist() == [-1.0, 0.5] and layer().tolist() == [-1.0, 0.5]
    assert isinstance(layer.lmbda, nn.Parameter) and "lmbda" in layer.state_dict()
    assert Multipliers(["x", "y"]).lmbda.tolist() == [0.0, 0.0]
    with pytest.raises(ValueError, match="names"):
        Multipliers({})


def test_logit_spans_the_line_and_comes_back_through_the_sigmoid():
    values = numpy.array([0.0, 0.25, 0.5, 0.75, 1.0])
    scaler = Logit(low=1e-3, high=1e-3)
    out = scaler.apply(values)
    assert out[2] == pytest.approx(0.0) and out[0] < -6.0 < 6.0 < out[4]
    assert scaler.inverse(out)[1] == pytest.approx(0.25)
    assert scaler.inverse_torch(torch.tensor(out))[3].item() == pytest.approx(0.75, abs=1e-6)
    saturated = scaler.inverse(numpy.array([-800.0, 800.0]))
    assert saturated[0] == 0.0 and saturated[1] == 1.0
    assert scaler.rescales and not scaler.fits
    with pytest.raises(ValueError):
        Logit(low=0.0)


def test_the_median_std_scaler_centers_on_the_median():
    values = numpy.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [100.0, 40.0]])
    scaler = MedianStdScaler()
    scaler.fit(values)
    assert scaler.center.tolist() == [2.5, 25.0] and scaler.grouped and scaler.fits
    out = scaler.apply(values)
    assert out[:, 0].tolist() == pytest.approx(((values[:, 0] - 2.5) / values[:, 0].std()).tolist())
    assert numpy.allclose(scaler.inverse(out), values)
    assert scaler.apply(values[:, 1], columns=[1]).tolist() == pytest.approx(out[:, 1].tolist())


def test_the_loader_takes_the_whole_set_and_drop_last_auto():
    dataset = table(frame(rows=9))
    whole = torch_loader(dataset, "train")
    assert len(whole) == 1 and next(iter(whole))["x"].shape == (9, 3)
    assert len(torch_loader(dataset, "valid", 4)) == 3 and len(torch_loader(dataset, "valid")) == 1
    assert len(torch_loader(dataset, "train", 4, drop_last="auto")) == 2
    assert len(torch_loader(dataset, "train", 3, drop_last="auto")) == 3
    assert len(torch_loader(table(frame(rows=0)), "valid")) == 0


def test_the_history_line_carries_the_seconds_and_the_git_note_says_which_code(tmp_path):
    optimizer = Sgd({"m": tiny_model()}, {"lr": 0.3}, None, "l")
    line = History.line({"train/l": 0.5}, {"turn": 2, "global_step": 8}, {"m": optimizer}, {}, 1.23456)
    assert line["seconds"] == 1.235 and list(line)[-2:] == ["seconds", "rules"]
    assert "seconds" not in History.line({}, {}, {}, {})
    named = History.line({}, {"turn": 1}, {"m": optimizer}, {}, minimizes={"m": "loss_mse"})
    assert named["minimizes/m"] == "loss_mse" and list(named)[-3:] == ["lr/m", "minimizes/m", "rules"]
    noted = History.line({}, {"turn": 1}, {"m": optimizer}, {}, effects={"l.terms.a": 1.0, "net.trainable": True})
    assert noted["effect/l.terms.a"] == 1.0 and noted["effect/net.trainable"] is True
    assert History([noted]).series() == {}
    assert git_note(tmp_path) == {"commit": None, "dirty": None}
    note = git_note(".")
    assert note["commit"] is None or len(note["commit"]) == 40
    write_git_note(tmp_path, [str(tmp_path / "cfg.yaml")])
    assert json.loads((tmp_path / "git.json").read_text())["commit"] is None
    with pytest.raises(Exception, match="final/state.pt"):
        weights_of(tmp_path, "final")
    (tmp_path / "final").mkdir()
    (tmp_path / "final" / "state.pt").write_bytes(b"")
    assert weights_of(tmp_path, "final").name == "state.pt"
