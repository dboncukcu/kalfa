import torch

from kalfa.registration import lego
from kalfa.std.optimizer.base import Optimizer, torch_factory


@lego("/optimizer/torch/adamw", state=True, refs={"loss": "loss", "schedule": "schedule"},
      aliases="models", alias="adamw", description="torch AdamW over the union of its models")
def adamw(models, params=None, schedule=None, loss=None):
    return Optimizer(torch_factory(torch.optim.AdamW), models, params, schedule, loss)
