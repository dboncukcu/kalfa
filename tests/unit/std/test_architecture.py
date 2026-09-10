"""kalfa's own architecture drawing: the graph, the traced shapes, the losses and the optimizers."""

import torch
from torch import nn

import kalfa  # noqa: F401
from helpers import frame, tiny_model
from kalfa.std.adapter.kalfa.criterion import CriterionAdapter
from kalfa.std.adapter.kalfa.objective import ObjectiveAdapter
from kalfa.std.criterion.kalfa.regression import mse
from kalfa.std.feed.kalfa.table import table
from kalfa.std.loader.kalfa.torch import torch_loader
from kalfa.std.optimizer.torch.optimizers import Sgd
from kalfa.std.plot.kalfa.architecture import architecture, graph_layout, traced_shapes, training_layout


def test_the_shapes_come_from_one_traced_batch():
    model = tiny_model()
    batch = next(iter(torch_loader(table(frame(rows=8)), "test", 4)))
    shapes = traced_shapes(model, batch, torch.device("cpu"))
    assert shapes["layer"] == ("2x3", "2x1") and shapes[None] == ("2x3", "2x1")
    assert traced_shapes(nn.Linear(3, 1), batch, torch.device("cpu")) == {}


def test_the_layout_follows_the_graph_and_attaches_the_training():
    model = tiny_model()
    layout, last = graph_layout(model, "m", {"layer": ("2x3", "2x1")})
    assert [box.kind for box in layout.boxes.values()] == ["input", "torch", "output"]
    assert layout.boxes["layer"].lines == ["layer", "Linear", "2x3 -> 2x1"] and last == 2
    assert layout.arrows == [("in:x", "layer", "x"), ("layer", "out:y", "y")]
    losses = {"loss_mse": CriterionAdapter(mse), "adv": ObjectiveAdapter(lambda models, batch: 0.0)}
    optimizers = {"m": Sgd({"m": model}, {"lr": 0.1}, None, "loss_mse"), "other": Sgd({}, {}, None, "adv")}
    training_layout(layout, model, "m", last, "m", losses, {"loss_mse": {"target": "price"}}, optimizers)
    assert layout.boxes["loss:loss_mse"].lines == ["loss_mse", "vs price"] and "loss:adv" not in layout.boxes
    assert layout.boxes["opt:m"].lines == ["m", "sgd lr 0.1"] and "opt:other" not in layout.boxes
    assert ("out:y", "loss:loss_mse", "") in layout.arrows and ("loss:loss_mse", "opt:m", "") in layout.arrows
    plain, last = graph_layout(nn.Linear(3, 1), "plain", {})
    assert list(plain.boxes) == ["plain"] and plain.boxes["plain"].lines == ["plain", "Linear"] and last == 0


def test_the_drawing_writes_one_file_per_model(tmp_path):
    model = tiny_model()
    loader = torch_loader(table(frame(rows=8)), "test", 4)
    architecture(None, [], {"m": model, "plain": nn.Linear(3, 1), "m.ema": model}, str(tmp_path),
                 loaders={"test": loader}, predicts="m", losses={"loss_mse": CriterionAdapter(mse)},
                 losses_keys={"loss_mse": {}}, optimizers={"m": Sgd({"m": model}, {"lr": 0.1}, None, "loss_mse")})
    written = sorted(path.name for path in (tmp_path / "plots").glob("*.png"))
    assert written == ["architecture_m.png", "architecture_plain.png"]
