from torch import nn

from kalfa.registration import lego


@lego("/layer/torch/relu", alias="relu", description="ReLU activation")
def relu():
    return nn.ReLU()
