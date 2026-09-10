import torch

from kalfa.registration import lego
from kalfa.std.optimizer.base import Optimizer


@lego("/optimizer/torch/sgd", state=True, refs={"loss": "loss", "schedule": "schedule"}, aliases="models",
      alias="sgd", description="torch SGD over the union of its models")
class Sgd(Optimizer):
    torch_class = torch.optim.SGD


@lego("/optimizer/torch/adam", state=True, refs={"loss": "loss", "schedule": "schedule"}, aliases="models",
      alias="adam", description="torch Adam over the union of its models; params are Adam's keyword arguments")
class Adam(Optimizer):
    torch_class = torch.optim.Adam


@lego("/optimizer/torch/adamw", state=True, refs={"loss": "loss", "schedule": "schedule"}, aliases="models",
      alias="adamw", description="torch AdamW over the union of its models")
class AdamW(Optimizer):
    torch_class = torch.optim.AdamW
