import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, inlets, outlets, embed):
        super().__init__()
        self.first = nn.Conv2d(inlets, outlets, 3, padding=1)
        self.time = nn.Linear(embed, outlets)
        self.norm = nn.GroupNorm(1, outlets)
        self.second = nn.Conv2d(outlets, outlets, 3, padding=1)

    def forward(self, x, embedded):
        hidden = torch.nn.functional.silu(self.first(x) + self.time(embedded)[:, :, None, None])
        return torch.nn.functional.silu(self.second(self.norm(hidden)))


class UNet(nn.Module):
    def __init__(self, channels=64, steps=1000, image_channels=3):
        super().__init__()
        self.embed = nn.Sequential(nn.Embedding(steps, channels), nn.SiLU())
        self.down = Block(image_channels, channels, channels)
        self.shrink = nn.Conv2d(channels, channels * 2, 4, stride=2, padding=1)
        self.middle = Block(channels * 2, channels * 2, channels)
        self.grow = nn.ConvTranspose2d(channels * 2, channels, 4, stride=2, padding=1)
        self.up = Block(channels * 2, channels, channels)
        self.out = nn.Conv2d(channels, image_channels, 3, padding=1)

    def forward(self, x, t):
        embedded = self.embed(t)
        skip = self.down(x, embedded)
        hidden = self.middle(self.shrink(skip), embedded)
        hidden = self.up(torch.cat([self.grow(hidden), skip], dim=1), embedded)
        return self.out(hidden)
