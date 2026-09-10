import torch
from torch import nn

from kalfa.registration import lego


@lego("/layer/kalfa/reparam", alias="reparam",
      description="Sample z from mu and logvar in train mode, return mu in eval mode")
class Reparam(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu, logvar):
        if not self.training:
            return mu
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
