from torch import nn

from kalfa.registration import lego


@lego("/layer/torch/leaky_relu", alias="leaky_relu", description="LeakyReLU activation")
def leaky_relu(negative_slope=0.01):
    return nn.LeakyReLU(negative_slope)
