"""The export kind."""

import pytest
import torch

import kalfa  # noqa: F401
from helpers import batch, tiny_model
from kalfa.std.export.kalfa.formats import onnx, state_dict, torchscript, traced_inputs


def test_state_dict_and_torchscript_write_the_model(tmp_path):
    model = tiny_model()
    inputs = traced_inputs(model, batch())
    assert len(inputs) == 1 and inputs[0].shape == (2, 3)
    with pytest.raises(KeyError):
        traced_inputs(model, {"y": torch.zeros(2)})
    saved = state_dict(model, inputs, tmp_path / "export", "m")
    assert saved.name == "m.pt" and set(torch.load(saved)) == set(model.state_dict())
    traced = torchscript(model, inputs, tmp_path / "scripted", "m")
    loaded = torch.jit.load(str(traced))
    assert torch.allclose(loaded(inputs[0]), model(inputs[0]))


def test_onnx_writes_the_graph_with_the_wire_names(tmp_path):
    pytest.importorskip("onnx")
    model = tiny_model()
    path = onnx(model, traced_inputs(model, batch()), tmp_path, "m")
    assert path.name == "m.onnx" and path.exists()
