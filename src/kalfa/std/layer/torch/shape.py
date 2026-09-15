import torch
from torch import nn


class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, *values):
        return torch.cat(values, dim=self.dim)


class LastStep(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, value):
        return value[:, -1, :]


def flatten(start_dim=1, end_dim=-1):
    return nn.Flatten(int(start_dim), int(end_dim))


def unflatten(dim, size):
    return nn.Unflatten(int(dim), tuple(int(part) for part in size))


def fold(output_size, kernel, dilation=1, padding=0, stride=1):
    return nn.Fold(output_size, kernel, dilation=dilation, padding=padding, stride=stride)


def unfold(kernel, dilation=1, padding=0, stride=1):
    return nn.Unfold(kernel, dilation=dilation, padding=padding, stride=stride)


def pixel_shuffle(factor):
    return nn.PixelShuffle(int(factor))


def pixel_unshuffle(factor):
    return nn.PixelUnshuffle(int(factor))


def channel_shuffle(groups):
    return nn.ChannelShuffle(int(groups))


def upsample(size=None, scale_factor=None, mode="nearest", align_corners=None):
    return nn.Upsample(size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)
