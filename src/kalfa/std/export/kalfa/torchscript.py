import torch

from kalfa.registration import lego
from kalfa.std.export.base import target_path


@lego("/export/kalfa/torchscript", alias="torchscript",
      description="The model traced with one batch and saved as <stem>.pt with torch.jit")
def torchscript(model, inputs, directory, stem):
    path = target_path(directory, stem, "pt")
    model.eval()
    with torch.no_grad():
        traced = torch.jit.trace(model, inputs)
    traced.save(str(path))
    return path
