import torch
from torch import nn

from kalfa.registration import lego


@lego("/layer/kalfa/l1_distance", alias="l1_distance",
      description="Mean absolute difference of two wires per sample")
class L1Distance(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, first, second):
        return (first - second).abs().reshape(first.shape[0], -1).mean(dim=1)


@lego("/layer/kalfa/reparam", alias="reparam",
      description="Sample z from mu and logvar in train mode, return mu in eval mode")
class Reparam(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu, logvar):
        if not self.training:
            return mu
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)


@lego("/layer/kalfa/unflatten", alias="unflatten",
      description="Reshape the features of every sample to shape")
def unflatten(shape):
    return nn.Unflatten(1, tuple(int(part) for part in shape))
