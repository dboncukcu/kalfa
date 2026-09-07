"""Test stand ins for a DCGAN plugin: a tiny generator and critic for 32 by 32 RGB images."""

import kalfa
import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, in_dim, channels):
        super().__init__()
        self.project = nn.Linear(int(in_dim), int(channels) * 4 * 4)
        self.channels = int(channels)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(self.channels, self.channels // 2, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(self.channels // 2, self.channels // 4, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(self.channels // 4, 3, 4, 2, 1), nn.Tanh())

    def forward(self, value):
        return self.up(self.project(value).reshape(len(value), self.channels, 4, 4))


class Critic(nn.Module):
    kalfa_lazy = True

    def __init__(self, channels, cond_dim):
        super().__init__()
        self.down = nn.Sequential(nn.LazyConv2d(int(channels) // 4, 4, 2, 1), nn.LeakyReLU(0.2),
                                  nn.Conv2d(int(channels) // 4, int(channels) // 2, 4, 2, 1), nn.LeakyReLU(0.2),
                                  nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.score = nn.Linear(int(channels) // 2 + int(cond_dim), 1)

    def forward(self, image, condition):
        return self.score(torch.cat([self.down(image), condition], dim=1))


@kalfa.lego("/layer/dcgan/generator", alias="dcgan_generator",
            description="Test generator: a linear projection and three transposed convolutions to 3 by 32 by 32")
def dcgan_generator(in_dim, channels):
    return Generator(in_dim, channels)


@kalfa.lego("/layer/dcgan/critic", alias="dcgan_critic",
            description="Test critic: two convolutions, global pooling, the condition concatenated, one score")
def dcgan_critic(channels, cond_dim):
    return Critic(channels, cond_dim)
