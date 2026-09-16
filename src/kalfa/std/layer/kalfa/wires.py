import torch
from torch import nn


class Add(nn.Module):
    def forward(self, *values):
        total = values[0]
        for value in values[1:]:
            total = total + value
        return total


class Subtract(nn.Module):
    def forward(self, first, second):
        return first - second


class Multiply(nn.Module):
    def forward(self, *values):
        total = values[0]
        for value in values[1:]:
            total = total * value
        return total


class Divide(nn.Module):
    def forward(self, first, second):
        return first / second


class Negate(nn.Module):
    def forward(self, value):
        return -value


class L1Distance(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, first, second):
        return (first - second).abs().reshape(first.shape[0], -1).mean(dim=1)


class Reparam(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu, logvar):
        if not self.training:
            return mu
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)


def unflatten(shape):
    return nn.Unflatten(1, tuple(int(part) for part in shape))
