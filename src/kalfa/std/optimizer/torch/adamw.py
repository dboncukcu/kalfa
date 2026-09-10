import torch

from kalfa.registration import lego
from kalfa.std.optimizer.base import Optimizer


@lego("/optimizer/torch/adamw", state=True, refs={"loss": "loss", "schedule": "schedule"}, aliases="models",
      alias="adamw", description="torch AdamW over the union of its models")
class AdamW(Optimizer):
    torch_class = torch.optim.AdamW
