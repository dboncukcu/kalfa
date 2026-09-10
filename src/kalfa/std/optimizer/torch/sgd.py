import torch

from kalfa.registration import lego
from kalfa.std.optimizer.base import Optimizer, torch_factory


@lego("/optimizer/torch/sgd", state=True, refs={"loss": "loss", "schedule": "schedule"},
      aliases="models", alias="sgd", description="torch SGD over the union of its models")
def sgd(models, params=None, schedule=None, loss=None):
    return Optimizer(torch_factory(torch.optim.SGD), models, params, schedule, loss)
