from torch import nn

from kalfa.registration import lego


@lego("/layer/torch/flatten", alias="flatten", description="Flatten every dimension but the batch")
def flatten():
    return nn.Flatten()
