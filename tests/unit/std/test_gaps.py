"""The small gaps: normalisation layers, kaiming uniform, weighted mse, feature width, the median std scaler,
the whole set as one batch, drop_last auto, the seconds of a turn, the git note and the final weights."""

import json

import numpy
import pytest
import torch
from torch import nn

import kalfa  # noqa: F401
from helpers import frame, tiny_model
from kalfa.api import git_note, weights_of, write_git_note
from kalfa.std.common.history import History
from kalfa.std.criterion.kalfa.weighted_mse import weighted_mse
from kalfa.std.data.kalfa.feature_width import feature_width
from kalfa.std.data.kalfa.target_weights import target_weights
from kalfa.std.feed.kalfa.table import table
from kalfa.std.init.torch.kaiming_uniform import kaiming_uniform
from kalfa.std.layer.torch.batch_norm import batch_norm
from kalfa.std.layer.torch.group_norm import group_norm
from kalfa.std.layer.torch.layer_norm import layer_norm
from kalfa.std.loader.kalfa.torch import torch_loader
from kalfa.std.optimizer.torch.sgd import Sgd
from kalfa.std.pre.kalfa.median_std_scaler import MedianStdScaler


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
