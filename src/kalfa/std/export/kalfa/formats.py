import inspect
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


def dynamic_batch(model, inputs):
    batch = torch.export.Dim("batch")
    specs = [{0: batch} if isinstance(value, torch.Tensor) else None for value in inputs]
    shapes = {}
    for parameter in inspect.signature(model.forward).parameters.values():
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            shapes[parameter.name] = tuple(specs)
            specs = []
        elif specs and parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            shapes[parameter.name] = specs.pop(0)
    return shapes


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


@lego("/export/kalfa/pt2", alias="pt2",
      description="The model exported with torch.export from one traced batch, the batch dimension left dynamic, "
                  "and saved as <stem>.pt2, the archive torch.export.load reads back")
def pt2(model, inputs, directory, stem):
    path = target_path(directory, stem, "pt2")
    model.eval()
    exported = torch.export.export(model, inputs, dynamic_shapes=dynamic_batch(model, inputs))
    torch.export.save(exported, str(path))
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
