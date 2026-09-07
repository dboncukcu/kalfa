"""Layers: legos that build nn.Module pieces for model graphs."""

import torch
from torch import nn

from ..registration import lego
from .deferred import later


class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, *values):
        return torch.cat(values, dim=self.dim)


@lego("/layer/kalfa/linear", alias="linear",
            description="Linear layer; without in_features the input width is taken from the first batch")
def linear(out_features, in_features=None):
    return later(_linear, out_features=out_features, in_features=in_features)


def _linear(out_features, in_features=None):
    if in_features is None:
        return nn.LazyLinear(int(out_features))
    return nn.Linear(int(in_features), int(out_features))


@lego("/layer/kalfa/linear_relu", alias="linear_relu",
            description="Linear layer followed by ReLU; lazy without in_features")
def linear_relu(out_features, in_features=None):
    return nn.Sequential(linear(out_features, in_features), nn.ReLU())


@lego("/layer/torch/linear", description="torch.nn.Linear")
def torch_linear(in_features, out_features):
    return later(_torch_linear, in_features=in_features, out_features=out_features)


def _torch_linear(in_features, out_features):
    return nn.Linear(int(in_features), int(out_features))


@lego("/layer/torch/concat", alias="concat", description="Concatenate wires along a dimension")
def concat(dim=1):
    return Concat(dim)


@lego("/layer/torch/flatten", alias="flatten", description="Flatten every dimension but the batch")
def flatten():
    return nn.Flatten()


@lego("/layer/torch/relu", alias="relu", description="ReLU activation")
def relu():
    return nn.ReLU()


@lego("/layer/torch/leaky_relu", alias="leaky_relu", description="LeakyReLU activation")
def leaky_relu(negative_slope=0.01):
    return nn.LeakyReLU(negative_slope)


@lego("/layer/torch/dropout", alias="dropout", description="Dropout")
def dropout(p=0.5):
    return nn.Dropout(p)


class L1Distance(nn.Module):
    def forward(self, first, second):
        return (first - second).abs().reshape(first.shape[0], -1).mean(dim=1)


@lego("/layer/kalfa/l1_distance", alias="l1_distance",
            description="Mean absolute difference of two wires per sample")
def l1_distance():
    return L1Distance()


class LazyGRU(nn.Module):
    """A GRU whose input width is taken from the first batch; returns the output sequence (batch, steps, hidden)."""

    kalfa_lazy = True

    def __init__(self, hidden, layers=1):
        super().__init__()
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.core = None

    def forward(self, value):
        if self.core is None:
            self.core = nn.GRU(value.shape[-1], self.hidden, num_layers=self.layers, batch_first=True).to(
                device=value.device, dtype=value.dtype)
        out, _ = self.core(value)
        return out


class LastStep(nn.Module):
    def forward(self, value):
        return value[:, -1, :]


@lego("/layer/torch/gru", alias="gru",
            description="GRU over (batch, steps, features) returning every step; the input width comes from the "
                        "first batch")
def gru(hidden, layers=1):
    return LazyGRU(hidden, layers)


@lego("/layer/torch/last_step", alias="last_step", description="The last step of a sequence")
def last_step():
    return LastStep()


@lego("/layer/kalfa/unflatten", alias="unflatten",
            description="Reshape the features of every sample to shape")
def unflatten(shape):
    return nn.Unflatten(1, tuple(int(part) for part in shape))


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


@lego("/layer/torch/embedding", alias="embedding",
            description="torch.nn.Embedding(num, dim); num may be a kind data component such as vocab_size")
def embedding(num, dim):
    return later(_embedding, num=num, dim=dim)


def _embedding(num, dim):
    return nn.Embedding(int(num), int(dim))


@lego("/layer/torch/conv2d", alias="conv2d",
            description="2d convolution; without in_channels the input channels are taken from the first batch")
def conv2d(out_channels, kernel, stride=1, padding=0, in_channels=None):
    if in_channels is None:
        return nn.LazyConv2d(int(out_channels), int(kernel), stride=int(stride), padding=int(padding))
    return nn.Conv2d(int(in_channels), int(out_channels), int(kernel), stride=int(stride), padding=int(padding))


@lego("/layer/torch/maxpool", alias="maxpool", description="2d max pooling")
def maxpool(kernel, stride=None):
    return nn.MaxPool2d(int(kernel), stride=None if stride is None else int(stride))
