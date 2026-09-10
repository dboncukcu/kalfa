import torch

from kalfa.registration import lego
from kalfa.std.criterion.base import pair


@lego("/criterion/kalfa/log_cosh", partial=True, alias="log_cosh",
      description="log(cosh(error)) loss")
def log_cosh(predictions, targets):
    predictions, targets = pair(predictions, targets)
    error = predictions - targets
    return (error + torch.nn.functional.softplus(-2.0 * error) - torch.log(torch.tensor(2.0))).mean()
