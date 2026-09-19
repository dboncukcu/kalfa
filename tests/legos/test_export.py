import sys

import pytest
import torch
from cirak.registry import registry

from helpers import batch, build, needs, tiny_model
from kalfa.std import STD_URIS
from kalfa.std.export.kalfa.formats import traced_inputs


EXPORT_URIS = sorted(uri for uri in STD_URIS if uri.startswith("/export/"))


def test_the_export_pack_holds_the_three_formats():
    assert EXPORT_URIS == ["/export/kalfa/onnx", "/export/kalfa/pt2", "/export/kalfa/state_dict"]
    assert registry.facts("/export/kalfa/onnx").get("requires") == "onnx"
    assert registry.facts("/export/kalfa/pt2").get("requires") is None


@pytest.mark.parametrize("uri", EXPORT_URIS, ids=[uri.rsplit("/", 1)[1] for uri in EXPORT_URIS])
def test_export_alias_is_its_name(uri):
    assert registry.aliases()[uri.rsplit("/", 1)[1]] == uri


def test_traced_inputs_take_the_first_rows_of_the_model_wires():
    model = tiny_model()
    fields = batch(rows=8)
    inputs = traced_inputs(model, fields)
    assert len(inputs) == 1 and tuple(inputs[0].shape) == (2, 3)
    assert torch.equal(inputs[0], fields["x"][:2])
    assert tuple(traced_inputs(model, fields, rows=3)[0].shape) == (3, 3)
    assert traced_inputs(model, {"x": "not a tensor"}) == ("not a tensor",)


def test_traced_inputs_name_the_missing_wire():
    with pytest.raises(KeyError) as caught:
        traced_inputs(tiny_model(), {"price": torch.zeros(2), "extra": torch.zeros(2)})
    assert caught.value.args[0] == "model input 'x' is not a batch field; the batch has ['extra', 'price']"


def test_state_dict_writes_the_plain_weights_as_stem_pt(tmp_path):
    model = tiny_model()
    inputs = traced_inputs(model, batch())
    path = build("/export/kalfa/state_dict", model=model, inputs=inputs, directory=tmp_path / "exports", stem="model0")
    assert path == tmp_path / "exports" / "model0.pt" and path.is_file()
    saved = torch.load(path)
    assert list(saved) == ["nodes.layer.weight", "nodes.layer.bias"]
    assert torch.equal(saved["nodes.layer.weight"], model.nodes["layer"].weight)
    assert torch.equal(saved["nodes.layer.bias"], model.nodes["layer"].bias)


def test_pt2_exports_the_traced_model_with_a_dynamic_batch(tmp_path):
    model = tiny_model(in_features=3, out_features=2, seed=4)
    model.train()
    inputs = traced_inputs(model, batch())
    path = build("/export/kalfa/pt2", model=model, inputs=inputs, directory=tmp_path / "exports", stem="model0")
    assert path == tmp_path / "exports" / "model0.pt2" and path.is_file()
    assert model.training is False
    loaded = torch.export.load(str(path)).module()
    assert torch.allclose(loaded(inputs[0]), model(inputs[0]))
    wider = batch(rows=8, seed=3)["x"]
    assert tuple(loaded(wider).shape) == (8, 2) and torch.allclose(loaded(wider), model(wider))
    single = batch(rows=1, seed=5)["x"]
    assert torch.allclose(loaded(single), model(single))


def test_onnx_exports_the_wires_as_the_graph_names(tmp_path):
    onnx = needs("onnx")
    model = tiny_model()
    inputs = traced_inputs(model, batch())
    path = build("/export/kalfa/onnx", model=model, inputs=inputs, directory=tmp_path / "exports", stem="model0")
    assert path == tmp_path / "exports" / "model0.onnx" and path.is_file()
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    assert [item.name for item in graph.graph.input] == ["x"]
    assert [item.name for item in graph.graph.output] == ["y"]
    assert graph.opset_import[0].version == 17
    older = build("/export/kalfa/onnx", model=model, inputs=inputs, directory=tmp_path / "exports", stem="older",
                  opset=13)
    assert onnx.load(str(older)).opset_import[0].version == 13


def test_onnx_is_skipped_without_the_library(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "onnx", None)
    model = tiny_model()
    inputs = traced_inputs(model, batch())
    skipped = build("/export/kalfa/onnx", model=model, inputs=inputs, directory=tmp_path / "exports", stem="model0")
    assert skipped is None and not (tmp_path / "exports").exists()
