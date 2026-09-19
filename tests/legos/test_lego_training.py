import json

import pytest
import torch
from cirak.registry import registry

from helpers import build, frame, tiny_model
from kalfa.std.lego.kalfa.clone import Ema
from kalfa.std.pre.base import Field, Prep


def fill(model, value):
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(value)
    return model


def weight_of(model):
    return model.nodes["layer"].weight


def test_clone_is_a_frozen_copy_of_a_ready_model():
    model = fill(tiny_model(), 1.0)
    ema = build("/lego/kalfa/clone", model=model, decay=0.9)
    assert isinstance(ema, Ema)
    assert registry.facts("/lego/kalfa/clone").state is True
    assert ema.decay == 0.9 and ema.inputs == ["x"] and ema.outputs == ["y"]
    assert ema.model is not model
    assert torch.equal(weight_of(ema.model), torch.ones(1, 3))
    assert ema.model.trainable is False
    assert all(parameter.requires_grad is False for parameter in ema.model.parameters())
    assert ema.training is False and ema.model.training is False
    ema.train()
    assert ema.training is False


def test_clone_shifts_towards_the_model_by_one_minus_decay():
    model = fill(tiny_model(), 1.0)
    ema = build("/lego/kalfa/clone", model=model, decay=0.9)
    fill(model, 2.0)
    ema.shift(model)
    assert torch.allclose(weight_of(ema.model), torch.full((1, 3), 1.1))
    ema.shift(model)
    assert torch.allclose(weight_of(ema.model), torch.full((1, 3), 1.19))
    assert torch.equal(weight_of(model), torch.full((1, 3), 2.0))
    x = torch.ones(2, 3)
    assert torch.allclose(ema(x), x @ weight_of(ema.model).t() + ema.model.nodes["layer"].bias)


def test_clone_state_dict_carries_the_copy_under_the_model_prefix():
    ema = build("/lego/kalfa/clone", model=fill(tiny_model(), 1.0), decay=0.5)
    state = ema.state_dict()
    assert sorted(state) == ["model.nodes.layer.bias", "model.nodes.layer.weight"]
    other = build("/lego/kalfa/clone", model=fill(tiny_model(), 3.0), decay=0.5)
    other.load_state_dict(state)
    assert torch.equal(weight_of(other.model), torch.ones(1, 3))
    assert all(parameter.requires_grad is False for parameter in other.model.parameters())
    assert other.load_state_dict({"unrelated": torch.zeros(1)}) is None


def test_clone_of_a_lazy_model_waits_for_its_first_batch():
    model = tiny_model(lazy=True)
    ema = build("/lego/kalfa/clone", model=model, decay=0.5)
    assert ema.model is None
    with pytest.raises(ValueError, match=r"the EMA copy has no weights yet: its model has not taken a batch"):
        ema(torch.ones(2, 3))
    ema.shift(model)
    assert ema.model is None
    model(torch.ones(2, 3))
    ema.shift(model)
    assert ema.model is not None
    assert torch.equal(weight_of(ema.model), weight_of(model))


def test_merge_prefixes_the_per_set_metrics_in_the_order_of_the_sets():
    parts = {"valid_metrics": {"rmse": 1.0, "mae": 0.5}, "train_metrics": {"mse": 2.0}, "test_metrics": {},
             "other": {"x": 1.0}}
    merged = build("/lego/kalfa/merge", parts=parts, prefixes={"train": "train", "valid": "val", "test": "test"})
    assert merged == {"train/mse": 2.0, "val/rmse": 1.0, "val/mae": 0.5, "other/x": 1.0}
    assert list(merged) == ["train/mse", "val/rmse", "val/mae", "other/x"]
    assert build("/lego/kalfa/merge", parts=None, prefixes={}) == {}


def test_architecture_note_writes_the_boxes_arrows_and_widths_of_every_model(tmp_path):
    data = frame(rows=8, set_name="test")
    prep = Prep([Field(name, [], False, [name]) for name in data.features] + [Field("price", [], True, ["price"])],
                {}, {}, {}, [])
    loader = build("/loader/kalfa/torch", data=build("/feed/kalfa/table", frame=data), set="test", eval_size=4)
    model = tiny_model()
    ema = build("/lego/kalfa/clone", model=model, decay=0.5)
    losses = {"mse": build("/adapter/kalfa/criterion", criterion=build("/criterion/kalfa/mse"))}
    optimizer = build("/optimizer/torch/sgd", models={"m": model}, params={"lr": 0.1}, loss="mse")
    result = build("/lego/kalfa/architecture_note", models={"m": model, "m.ema": ema}, loaders={"test": loader},
                   prep=prep, predicts="m", losses=losses, losses_keys={"mse": {"output": "y", "target": "price"}},
                   optimizers={"main": optimizer}, record=str(tmp_path))
    assert result is None
    note = json.loads((tmp_path / "architecture.json").read_text())
    assert set(note) == {"models", "features", "targets"}
    assert note["features"] == ["x0", "x1", "x2"]
    assert note["targets"] == {"price": ["price"]}
    assert list(note["models"]) == ["m"]
    drawn = note["models"]["m"]
    assert drawn["parameters"] == 4 and drawn["trainable"] == 4
    boxes = {box["name"]: box for box in drawn["boxes"]}
    assert list(boxes) == ["in:x", "layer", "out:y", "loss:mse", "opt:main"]
    assert [box["kind"] for box in drawn["boxes"]] == ["input", "torch", "output", "loss", "optimizer"]
    assert boxes["in:x"]["lines"] == ["x", "3 features", "x0, x1, x2"]
    assert boxes["layer"]["lines"] == ["layer", "Linear", "2x3 -> 2x1"]
    assert boxes["layer"]["parameters"] == 4
    assert boxes["loss:mse"]["lines"] == ["mse", "vs price"]
    assert boxes["opt:main"]["lines"] == ["main", "sgd lr 0.1"]
    assert drawn["arrows"] == [["in:x", "layer", "x"], ["layer", "out:y", "y"], ["out:y", "loss:mse", ""],
                               ["loss:mse", "opt:main", ""]]
    assert drawn["widths"] == {"x": "3", "y": "1"}


def test_architecture_note_without_a_record_writes_nothing(tmp_path):
    assert build("/lego/kalfa/architecture_note", models={"m": tiny_model()}, loaders={}) is None
    assert list(tmp_path.iterdir()) == []
