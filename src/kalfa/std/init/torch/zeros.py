from torch import nn

from kalfa.registration import lego


@lego("/init/torch/zeros", alias="zeros", description="Zero initialization")
def zeros():
    def apply(tensor):
        nn.init.zeros_(tensor)
    return apply
