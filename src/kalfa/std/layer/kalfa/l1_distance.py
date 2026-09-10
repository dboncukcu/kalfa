from torch import nn

from kalfa.registration import lego


class L1Distance(nn.Module):
    def forward(self, first, second):
        return (first - second).abs().reshape(first.shape[0], -1).mean(dim=1)


@lego("/layer/kalfa/l1_distance", alias="l1_distance",
      description="Mean absolute difference of two wires per sample")
def l1_distance():
    return L1Distance()
