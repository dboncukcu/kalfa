"""Inits: legos that return a tensor initializer, applied per role by the builder."""


from torch import nn

from ..registration import lego


@lego("/init/torch/normal", alias="normal", description="Normal initialization with std and mean")
def normal(std, mean=0.0):
    def apply(tensor):
        nn.init.normal_(tensor, mean=mean, std=std)
    return apply


@lego("/init/torch/xavier", alias="xavier", description="Xavier uniform initialization")
def xavier(gain=1.0):
    def apply(tensor):
        nn.init.xavier_uniform_(tensor, gain=gain)
    return apply


@lego("/init/torch/kaiming", alias="kaiming", description="Kaiming normal initialization")
def kaiming(nonlinearity="relu"):
    def apply(tensor):
        nn.init.kaiming_normal_(tensor, nonlinearity=nonlinearity)
    return apply


@lego("/init/torch/zeros", alias="zeros", description="Zero initialization")
def zeros():
    def apply(tensor):
        nn.init.zeros_(tensor)
    return apply
