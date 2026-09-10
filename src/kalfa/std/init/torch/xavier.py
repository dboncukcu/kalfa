from torch import nn

from kalfa.registration import lego


@lego("/init/torch/xavier", alias="xavier", description="Xavier uniform initialization")
def xavier(gain=1.0):
    def apply(tensor):
        nn.init.xavier_uniform_(tensor, gain=gain)
    return apply
