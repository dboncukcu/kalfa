import torch

from kalfa.registration import lego
from kalfa.std.criterion.base import pair


@lego("/criterion/kalfa/huber", partial=True, alias="huber",
      description="Huber loss with threshold delta")
def huber(predictions, targets, delta=1.0):
    predictions, targets = pair(predictions, targets)
    return torch.nn.functional.huber_loss(predictions, targets, delta=delta)
