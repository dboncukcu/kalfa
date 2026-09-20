import json

import pytest
import torch
from cirak.build import Graph, GraphNode
from cirak.registry import registry
from torch import nn

from helpers import build, frame, tiny_model
from kalfa.std import STD_URIS
from kalfa.std.builder.kalfa.module import Module
from kalfa.std.checkpoint.base import load, payload, save
from kalfa.std.lego.kalfa.clone import Ema
from kalfa.std.pre.base import Field, Prep


MODEL_LEGOS = ["/lego/kalfa/architecture_note", "/lego/kalfa/clone", "/lego/kalfa/const", "/lego/kalfa/identity",
               "/lego/kalfa/merge", "/lego/kalfa/pack", "/lego/kalfa/pixel_features", "/lego/kalfa/select"]
EMA_MESSAGE = "the EMA copy has no weights yet: its model has not taken a batch"


def norm_model(seed=1):
    graph = Graph(("x",), ("y",), (GraphNode("norm", nn.BatchNorm1d(3), ("x",), ("y",)),))
    return Module(graph, seed=seed, name="norm")


def block_model(seed=1):
    block = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 1))
    return Module(Graph(("x",), ("y",), (GraphNode("block", block, ("x",), ("y",)),)), seed=seed, name="m")


def loader_of(rows=8, size=4):
    dataset = build("/feed/kalfa/table", frame=frame(rows=rows))
    return build("/loader/kalfa/torch", data=dataset, set="test", size=size)


def box(name, kind, lines, column, detail=(), layers=(), parameters=0):
    return {"name": name, "kind": kind, "lines": lines, "column": column, "row": 0, "detail": list(detail),
            "layers": list(layers), "parameters": parameters}


def test_the_model_legos_are_registered_with_their_facts():
    assert set(MODEL_LEGOS) <= STD_URIS
    assert all(registry.facts(uri).kind == "lego" for uri in MODEL_LEGOS)
    assert registry.facts("/lego/kalfa/clone").state is True
    assert registry.facts("/lego/kalfa/select").returns == "selected"
    assert registry.facts("/lego/kalfa/select").bus == {"record": "record"}
    assert registry.facts("/lego/kalfa/architecture_note").returns is None
    assert registry.facts("/lego/kalfa/architecture_note").bus == {"record": "record", "device": "device"}
    assert registry.facts("/lego/kalfa/pack").aliases == ("items",)
    assert registry.facts("/lego/kalfa/identity").aliases == ("value",)
    assert registry.facts("/lego/kalfa/const").aliases == ()
    assert registry.facts("/lego/kalfa/merge").aliases == ()
    assert not any(uri in registry.aliases().values() for uri in MODEL_LEGOS)


def test_clone_is_a_frozen_copy_of_the_model_in_eval_mode():
    model = tiny_model(seed=1)
    model.train()
    ema = build("/lego/kalfa/clone", model=model, decay=0.9)
    assert isinstance(ema, Ema) and ema.decay == 0.9 and ema.source is model
    assert ema.inputs == ["x"] and ema.outputs == ["y"]
    assert ema.model is not model and ema.model.trainable is False and ema.model.training is False
    assert torch.equal(ema.model.nodes["layer"].weight, model.nodes["layer"].weight)
    assert ema.training is False and ema.train() is ema and ema.training is False
    assert all(not parameter.requires_grad for parameter in ema.parameters())
    assert all(parameter.requires_grad for parameter in model.parameters()) and model.training
    x = torch.ones(2, 3)
    assert torch.equal(ema(x), ema.model(x)) and torch.equal(ema(x), model(x))


def test_clone_shifts_towards_the_model_by_its_decay_and_copies_the_buffers():
    model = tiny_model(seed=1)
    ema = build("/lego/kalfa/clone", model=model, decay=0.5)
    before = ema.model.nodes["layer"].weight.detach().clone()
    with torch.no_grad():
        model.nodes["layer"].weight.fill_(1.0)
        model.nodes["layer"].bias.fill_(3.0)
    ema.shift(model)
    assert torch.allclose(ema.model.nodes["layer"].weight, 0.5 * before + 0.5)
    ema.shift(model)
    assert torch.allclose(ema.model.nodes["layer"].weight, 0.25 * before + 0.75)
    assert torch.equal(model.nodes["layer"].weight, torch.ones(1, 3))
    normed = norm_model()
    copy = build("/lego/kalfa/clone", model=normed, decay=0.5)
    normed.train()
    normed(torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [2.0, 2.0, 2.0], [0.0, 4.0, 2.0]]))
    assert not torch.equal(copy.model.nodes["norm"].running_mean, normed.nodes["norm"].running_mean)
    copy.shift(normed)
    assert torch.equal(copy.model.nodes["norm"].running_mean, normed.nodes["norm"].running_mean)
    assert torch.equal(copy.model.nodes["norm"].running_var, normed.nodes["norm"].running_var)
    assert int(copy.model.nodes["norm"].num_batches_tracked) == 1


def test_clone_state_dict_carries_the_copy_under_the_model_prefix():
    model = tiny_model(seed=1)
    ema = build("/lego/kalfa/clone", model=model, decay=0.9)
    assert list(ema.state_dict()) == ["model.nodes.layer.weight", "model.nodes.layer.bias"]
    other = tiny_model(seed=5)
    state = {f"model.{key}": value for key, value in other.state_dict().items()}
    result = ema.load_state_dict(state)
    assert result.missing_keys == [] and result.unexpected_keys == []
    assert torch.equal(ema.model.nodes["layer"].weight, other.nodes["layer"].weight)
    assert all(not parameter.requires_grad for parameter in ema.parameters()) and ema.model.training is False
    assert ema.load_state_dict({"unrelated": torch.zeros(1)}) is None
    assert torch.equal(ema.model.nodes["layer"].weight, other.nodes["layer"].weight)
    with pytest.raises(RuntimeError):
        ema.load_state_dict({"model.nodes.layer.weight": torch.zeros(1, 3)})


def test_clone_of_a_lazy_model_waits_for_its_first_batch():
    source = tiny_model(seed=2, lazy=True)
    ema = build("/lego/kalfa/clone", model=source, decay=0.9)
    assert ema.model is None and ema.inputs == ["x"]
    with pytest.raises(ValueError) as caught:
        ema(torch.ones(2, 4))
    assert str(caught.value) == EMA_MESSAGE
    ema.shift(source)
    assert ema.model is None
    x = torch.ones(2, 4)
    source(x)
    ema.shift(source)
    assert ema.model is not None and ema.model is not source and ema.model.training is False
    assert torch.equal(ema.model.nodes["layer"].weight, source.nodes["layer"].weight)
    assert torch.equal(ema(x), source(x))
    late = build("/lego/kalfa/clone", model=tiny_model(seed=3, lazy=True), decay=0.9)
    late.source(x)
    late.load_state_dict({f"model.{key}": value for key, value in source.state_dict().items()})
    assert torch.equal(late.model.nodes["layer"].weight, source.nodes["layer"].weight)
    assert all(not parameter.requires_grad for parameter in late.parameters())


def test_pack_maps_the_given_items():
    items = {"a": 1, "b": [2]}
    packed = build("/lego/kalfa/pack", items=items)
    assert packed == items and packed is not items and packed["b"] is items["b"]
    assert build("/lego/kalfa/pack", items=None) == {}
    assert build("/lego/kalfa/pack", items=[("k", "v")]) == {"k": "v"}


def test_const_hands_out_a_fresh_copy_and_identity_the_value_itself():
    value = {"lr": [1.0, 2.0], "name": "x"}
    copied = build("/lego/kalfa/const", value=value)
    assert copied == value and copied is not value and copied["lr"] is not value["lr"]
    assert build("/lego/kalfa/const", value=3) == 3 and build("/lego/kalfa/const", value=None) is None
    assert build("/lego/kalfa/identity", value=value) is value
    assert build("/lego/kalfa/identity", value=None) is None


def test_merge_prefixes_the_per_set_metrics_in_the_order_of_the_prefixes():
    parts = {"test_metrics": {"acc": 0.5}, "valid_metrics": {"loss": 2.0, "acc": 0.4}, "train_metrics": {"loss": 1.0}}
    prefixes = {"train": "train", "valid": "val", "test": "test"}
    merged = build("/lego/kalfa/merge", parts=parts, prefixes=prefixes)
    assert merged == {"train/loss": 1.0, "val/loss": 2.0, "val/acc": 0.4, "test/acc": 0.5}
    assert list(merged) == ["train/loss", "val/loss", "val/acc", "test/acc"]
    extra = build("/lego/kalfa/merge", parts={"holdout_metrics": {"mae": 3.0}, "train_metrics": {"mae": 1.0},
                                              "empty_metrics": None}, prefixes=prefixes)
    assert list(extra.items()) == [("train/mae", 1.0), ("holdout/mae", 3.0)]
    assert build("/lego/kalfa/merge", parts=None, prefixes=prefixes) == {}
    assert build("/lego/kalfa/merge", parts={"train": {"a": 1}}, prefixes={"train": "tr"}) == {"tr/a": 1}


def test_select_best_loads_the_best_checkpoint_into_copies(tmp_path):
    model = tiny_model(seed=1)
    ema = build("/lego/kalfa/clone", model=model, decay=0.9)
    best_model, best_ema = tiny_model(seed=7), build("/lego/kalfa/clone", model=tiny_model(seed=8), decay=0.9)
    record = tmp_path / "run"
    save(record / "checkpoints" / "best.pt", payload({"m": best_model}, {}, {"m": best_ema}, {"turn": 3}, {}))
    assert load(record / "checkpoints" / "best.pt")["turn"] == 3
    model.train()
    selected = build("/lego/kalfa/select", models={"m": model}, emas={"m": ema}, which="best", record=str(record))
    assert list(selected) == ["m", "m.ema"] and isinstance(selected["m.ema"], Ema)
    assert torch.equal(selected["m"].nodes["layer"].weight, best_model.nodes["layer"].weight)
    assert torch.equal(selected["m"].nodes["layer"].bias, best_model.nodes["layer"].bias)
    assert torch.equal(selected["m.ema"].model.nodes["layer"].weight, best_ema.model.nodes["layer"].weight)
    assert selected["m"].training is False and selected["m.ema"].training is False
    assert selected["m"].inputs == ["x"] and torch.equal(selected["m"](torch.ones(2, 3)), best_model(torch.ones(2, 3)))
    assert all(not parameter.requires_grad for parameter in selected["m.ema"].parameters())


def test_select_last_copies_the_models_as_training_left_them(tmp_path):
    model = tiny_model(seed=1)
    model.train()
    before = model.nodes["layer"].weight.detach().clone()
    selected = build("/lego/kalfa/select", models={"m": model}, emas=None, which="last", record=str(tmp_path))
    assert list(selected) == ["m"] and selected["m"].training is False
    assert torch.equal(selected["m"].nodes["layer"].weight, before)
    assert not (tmp_path / "checkpoints").exists()
    with_ema = build("/lego/kalfa/select", models={"m": model},
                     emas={"m": build("/lego/kalfa/clone", model=model, decay=0.5)}, which="last", record=str(tmp_path))
    assert list(with_ema) == ["m", "m.ema"]


def test_select_refuses_a_missing_best_checkpoint_and_another_report(tmp_path):
    models = {"m": tiny_model(seed=1)}
    with pytest.raises(FileNotFoundError) as caught:
        build("/lego/kalfa/select", models=models, emas={}, which="best", record=str(tmp_path))
    assert str(caught.value) == (f"report: best needs {tmp_path / 'checkpoints' / 'best.pt'}, but the best checkpoint "
                                 f"was never written")
    with pytest.raises(ValueError) as caught:
        build("/lego/kalfa/select", models=models, emas={}, which="final", record=str(tmp_path))
    assert str(caught.value) == "report must be best or last, got 'final'"


def test_architecture_note_writes_the_boxes_arrows_and_widths_of_every_report_model(tmp_path):
    model = tiny_model(seed=1)
    ema = build("/lego/kalfa/clone", model=model, decay=0.9)
    result = build("/lego/kalfa/architecture_note", models={"m": model, "m.ema": ema}, loaders={"test": loader_of()},
                   record=str(tmp_path))
    assert result is None
    note = json.loads((tmp_path / "architecture.json").read_text())
    assert set(note) == {"models", "features", "targets"} and note["features"] == [] and note["targets"] == {}
    assert list(note["models"]) == ["m"]
    assert set(note["models"]["m"]) == {"boxes", "arrows", "widths", "parameters", "trainable"}
    assert note["models"]["m"]["boxes"] == [box("in:x", "input", ["x"], 0),
                                            box("layer", "torch", ["layer", "Linear", "2x3 -> 2x1"], 1, parameters=4),
                                            box("out:y", "output", ["y"], 2)]
    assert note["models"]["m"]["arrows"] == [["in:x", "layer", "x"], ["layer", "out:y", "y"]]
    assert note["models"]["m"]["widths"] == {"x": "3", "y": "1"}
    assert note["models"]["m"]["parameters"] == 4 and note["models"]["m"]["trainable"] == 4


def test_architecture_note_lists_the_features_and_the_layers_inside_a_node(tmp_path):
    model = block_model()
    fields = [Field(name, [], False, [name]) for name in ("x0", "x1", "x2")] + [Field("price", [], True, ["price"])]
    prep = Prep(fields, {}, {}, {}, [])
    build("/lego/kalfa/architecture_note", models={"m": model}, loaders={"valid": loader_of()}, prep=prep,
          record=str(tmp_path))
    note = json.loads((tmp_path / "architecture.json").read_text())
    assert note["features"] == ["x0", "x1", "x2"] and note["targets"] == {"price": ["price"]}
    boxes = {item["name"]: item for item in note["models"]["m"]["boxes"]}
    assert boxes["in:x"] == box("in:x", "input", ["x", "3 features", "x0, x1, x2"], 0, detail=["x0", "x1", "x2"])
    assert boxes["block"]["lines"] == ["block", "Sequential", "2x3 -> 2x1"] and boxes["block"]["parameters"] == 21
    assert boxes["block"]["layers"] == [
        {"name": "0", "kind": "torch", "class": "Linear", "summary": "3, 4", "shapes": ["2x3", "2x4"], "parameters": 16,
         "children": []},
        {"name": "1", "kind": "torch", "class": "ReLU", "summary": "", "shapes": ["2x4", "2x4"], "parameters": 0,
         "children": []},
        {"name": "2", "kind": "torch", "class": "Linear", "summary": "4, 1", "shapes": ["2x4", "2x1"], "parameters": 5,
         "children": []}]
    assert note["models"]["m"]["parameters"] == 21 and note["models"]["m"]["trainable"] == 21


def test_architecture_note_without_a_batch_or_a_graph_still_draws_the_boxes(tmp_path):
    frozen = tiny_model(seed=1, trainable=False)
    build("/lego/kalfa/architecture_note", models={"m": frozen, "plain": nn.Linear(3, 1)}, loaders={},
          record=str(tmp_path))
    note = json.loads((tmp_path / "architecture.json").read_text())
    assert list(note["models"]) == ["m", "plain"]
    assert note["models"]["m"]["boxes"][1] == box("layer", "torch", ["layer", "Linear"], 1, parameters=4)
    assert note["models"]["m"]["widths"] == {} and note["models"]["m"]["trainable"] == 0
    assert note["models"]["plain"]["boxes"] == [box("plain", "lego", ["plain", "Linear"], 0, parameters=4)]
    assert note["models"]["plain"]["arrows"] == [] and note["models"]["plain"]["widths"] == {}
    assert build("/lego/kalfa/architecture_note", models={"m": frozen}, loaders={}, record=None) is None
    assert not (tmp_path / "other").exists()


def test_architecture_note_draws_composites_as_model_boxes(tmp_path):
    encoder = tiny_model(in_features=3, out_features=2, seed=1)
    graph = Graph(("x",), ("s",), (GraphNode("z", None, ("x",), ("z",), ref="encoder"),
                                   GraphNode("s", nn.Identity(), ("z",), ("s",))))
    composite = Module(graph, models={"encoder": encoder}, name="full")
    build("/lego/kalfa/architecture_note", models={"encoder": encoder}, loaders={"test": loader_of()},
          composites={"full": composite}, record=str(tmp_path))
    note = json.loads((tmp_path / "architecture.json").read_text())
    assert list(note["models"]) == ["encoder", "full"]
    boxes = {item["name"]: item for item in note["models"]["full"]["boxes"]}
    assert boxes["z"] == box("z", "model", ["z", "model encoder", "2x3 -> 2x2"], 1, parameters=8)
    assert boxes["s"] == box("s", "torch", ["s", "Identity", "2x2 -> 2x2"], 2)
    assert note["models"]["full"]["arrows"] == [["in:x", "z", "x"], ["z", "s", "z"], ["s", "out:s", "s"]]
    assert note["models"]["full"]["widths"] == {"x": "3", "z": "2", "s": "2"}
    assert note["models"]["full"]["parameters"] == note["models"]["encoder"]["parameters"] == 8


def test_pixel_features_pool_every_image_to_size_by_size_and_flatten():
    extract = build("/lego/kalfa/pixel_features", size=2)
    images = torch.arange(16.0).reshape(1, 1, 4, 4)
    assert extract(images).tolist() == [[2.5, 4.5, 10.5, 12.5]]
    colour = torch.stack([torch.zeros(4, 4), torch.ones(4, 4), torch.full((4, 4), 2.0)]).reshape(1, 3, 4, 4)
    assert extract(colour).tolist() == [[0.0] * 4 + [1.0] * 4 + [2.0] * 4]
    default = build("/lego/kalfa/pixel_features")
    assert tuple(default(torch.rand(2, 3, 8, 8)).shape) == (2, 48)
    assert torch.equal(default(images), images.reshape(1, 16))
