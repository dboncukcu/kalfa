"""Test stand in for a UNet plugin: a tiny noise predictor taking the image and the time step."""

import math

import kalfa
import torch
from torch import nn


class TinyUnet(nn.Module):
    kalfa_lazy = True

    def __init__(self, channels):
        super().__init__()
        self.channels = int(channels)
        self.first = nn.LazyConv2d(self.channels, 3, padding=1)
        self.time = nn.Linear(16, self.channels)
        self.last = nn.Conv2d(self.channels, 3, 3, padding=1)

    def embed(self, t):
        positions = torch.arange(8, device=t.device, dtype=torch.float32)
        angles = t.float()[:, None] / (1000.0 ** (positions[None, :] / 8.0))
        return torch.cat([angles.sin(), angles.cos()], dim=1)

    def forward(self, image, t):
        hidden = torch.relu(self.first(image) + self.time(self.embed(t))[:, :, None, None])
        return self.last(hidden)


@kalfa.lego("/layer/unet/unet", alias="unet",
            description="Test noise predictor: two convolutions with a sinusoidal time embedding in between")
def unet(channels):
    return TinyUnet(channels)
