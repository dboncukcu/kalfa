import torch

from kalfa.registration import lego
from kalfa.std.export.base import target_path


@lego("/export/kalfa/state_dict", alias="state_dict",
      description="The model's state_dict as <stem>.pt, the plain torch weights")
def state_dict(model, inputs, directory, stem):
    path = target_path(directory, stem, "pt")
    torch.save(model.state_dict(), path)
    return path
