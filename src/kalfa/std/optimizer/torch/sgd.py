import torch

from kalfa.registration import lego
from kalfa.std.optimizer.base import Optimizer


@lego("/optimizer/torch/sgd", state=True, refs={"loss": "loss", "schedule": "schedule"}, aliases="models",
      alias="sgd", description="torch SGD over the union of its models")
class Sgd(Optimizer):
    torch_class = torch.optim.SGD
