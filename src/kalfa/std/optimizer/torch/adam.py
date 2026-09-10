import torch

from kalfa.registration import lego
from kalfa.std.optimizer.base import Optimizer, torch_factory


@lego("/optimizer/torch/adam", state=True, refs={"loss": "loss", "schedule": "schedule"},
      aliases="models", alias="adam", description="torch Adam over the union of its models; params are Adam's keyword arguments")
def adam(models, params=None, schedule=None, loss=None):
    return Optimizer(torch_factory(torch.optim.Adam), models, params, schedule, loss)
