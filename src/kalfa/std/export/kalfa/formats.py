from pathlib import Path

import torch

from kalfa.registration import lego
from kalfa.std.common.optional import load


def traced_inputs(model, batch, rows=2):
    inputs = []
    for wire in model.inputs:
        if wire not in batch:
            raise KeyError(f"model input {wire!r} is not a batch field; the batch has {sorted(batch)}")
        value = batch[wire]
        inputs.append(value[:rows] if isinstance(value, torch.Tensor) else value)
    return tuple(inputs)


def target_path(directory, stem, suffix):
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{stem}.{suffix}"


@lego("/export/kalfa/state_dict", alias="state_dict",
      description="The model's state_dict as <stem>.pt, the plain torch weights")
def state_dict(model, inputs, directory, stem):
    path = target_path(directory, stem, "pt")
    torch.save(model.state_dict(), path)
    return path


@lego("/export/kalfa/torchscript", alias="torchscript",
      description="The model traced with one batch and saved as <stem>.pt with torch.jit")
def torchscript(model, inputs, directory, stem):
    path = target_path(directory, stem, "pt")
    model.eval()
    with torch.no_grad():
        traced = torch.jit.trace(model, inputs)
    traced.save(str(path))
    return path


@lego("/export/kalfa/onnx", alias="onnx", requires="onnx",
      description="The model exported to <stem>.onnx from one traced batch, the wires as the input and output "
                  "names, at the opset given")
def onnx(model, inputs, directory, stem, opset=17):
    if load("onnx", "the onnx export") is None:
        return None
    path = target_path(directory, stem, "onnx")
    model.eval()
    with torch.no_grad():
        torch.onnx.export(model, inputs, str(path), input_names=list(model.inputs), output_names=list(model.outputs),
                          opset_version=int(opset), dynamo=False)
    return path
