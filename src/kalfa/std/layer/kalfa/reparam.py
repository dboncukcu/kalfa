import torch
from torch import nn

from kalfa.registration import lego


class Reparam(nn.Module):
    """The VAE reparameterization: a sample of N(mu, exp(logvar)) in train mode, mu in eval mode."""

    def forward(self, mu, logvar):
        if not self.training:
            return mu
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)


@lego("/layer/kalfa/reparam", alias="reparam",
      description="Sample z from mu and logvar in train mode, return mu in eval mode")
def reparam():
    return Reparam()
