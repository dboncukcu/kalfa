from torch import nn

from kalfa.registration import lego


@lego("/init/torch/kaiming_uniform", alias="kaiming_uniform", description="Kaiming uniform initialization")
def kaiming_uniform(nonlinearity="relu"):
    def apply(tensor):
        nn.init.kaiming_uniform_(tensor, nonlinearity=nonlinearity)
    return apply
