from torch import nn

from kalfa.registration import lego


@lego("/layer/kalfa/l1_distance", alias="l1_distance",
      description="Mean absolute difference of two wires per sample")
class L1Distance(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, first, second):
        return (first - second).abs().reshape(first.shape[0], -1).mean(dim=1)
