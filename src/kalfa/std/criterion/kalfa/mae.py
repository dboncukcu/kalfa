from kalfa.registration import lego
from kalfa.std.criterion.base import pair


@lego("/criterion/kalfa/mae", partial=True, alias="mae", description="Mean absolute error")
def mae(predictions, targets):
    predictions, targets = pair(predictions, targets)
    return (predictions - targets).abs().mean()
