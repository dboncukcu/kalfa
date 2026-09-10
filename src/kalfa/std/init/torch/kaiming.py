from torch import nn

from kalfa.registration import lego


@lego("/init/torch/kaiming", alias="kaiming", description="Kaiming normal initialization")
def kaiming(nonlinearity="relu"):
    def apply(tensor):
        nn.init.kaiming_normal_(tensor, nonlinearity=nonlinearity)
    return apply
