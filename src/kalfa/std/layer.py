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


class Polynomial(nn.Module):
    """The polynomial expansion of a feature vector: the features themselves and every product of ``degree`` of
    them. The index list is built on the first batch, when the width is known."""

    def __init__(self, degree, interaction_only, bias, keep):
        super().__init__()
        self.degree = int(degree)
        self.interaction_only = bool(interaction_only)
        self.bias = bool(bias)
        self.keep = bool(keep)
        if self.degree < 2:
            raise ValueError(f"degree must be 2 or more, got {degree!r}")
        self.index = None

    def _indices(self, width, device):
        """One index table per order: an order k table holds the k column positions of every product of order k."""
        from itertools import combinations, combinations_with_replacement

        pick = combinations if self.interaction_only else combinations_with_replacement
        tables = []
        for order in range(2, self.degree + 1):
            rows = [list(entry) for entry in pick(range(width), order)]
            if rows:
                tables.append(torch.tensor(rows, dtype=torch.long, device=device))
        return tables

    def forward(self, values):
        flat = values.reshape(values.shape[0], -1)
        if self.index is None or not self.index or self.index[0].device != flat.device:
            self.index = self._indices(flat.shape[1], flat.device)
        parts = [flat] if self.keep else []
        parts.extend(flat[:, table].prod(dim=2) for table in self.index)
        if self.bias:
            parts.insert(0, torch.ones(flat.shape[0], 1, dtype=flat.dtype, device=flat.device))
        return torch.cat(parts, dim=1)


@lego("/layer/kalfa/polynomial", alias="polynomial",
            description="Polynomial expansion of the feature vector: the features and every product of degree of "
                        "them (interaction_only drops the squares, bias adds a constant column, keep: false "
                        "returns the products alone); the place for feature interactions, computed per batch")
def polynomial(degree=2, interaction_only=False, bias=False, keep=True):
    return Polynomial(degree, interaction_only, bias, keep)


class L2Normalize(nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = float(eps)

    def forward(self, values):
        flat = values.reshape(values.shape[0], -1)
        return flat / flat.norm(dim=1, keepdim=True).clamp_min(self.eps)


@lego("/layer/kalfa/l2_normalize", alias="l2_normalize",
            description="Divide every sample by the L2 norm of its own feature vector (sklearn's Normalizer as a "
                        "layer: it reads the whole vector, so it belongs to the model, not to a column chain)")
def l2_normalize(eps=1e-12):
    return L2Normalize(eps)


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
