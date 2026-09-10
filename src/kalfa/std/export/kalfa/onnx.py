import torch

from kalfa.registration import lego
from kalfa.std.common.optional import load
from kalfa.std.export.base import target_path


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
