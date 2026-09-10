from torch import nn

from kalfa.registration import lego


@lego("/init/torch/normal", alias="normal", description="Normal initialization with std and mean")
def normal(std, mean=0.0):
    def apply(tensor):
        nn.init.normal_(tensor, mean=mean, std=std)
    return apply
