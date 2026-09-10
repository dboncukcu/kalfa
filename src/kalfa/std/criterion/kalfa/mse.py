from kalfa.registration import lego
from kalfa.std.criterion.base import pair


@lego("/criterion/kalfa/mse", partial=True, alias="mse", description="Mean squared error")
def mse(predictions, targets):
    predictions, targets = pair(predictions, targets)
    return ((predictions - targets) ** 2).mean()
