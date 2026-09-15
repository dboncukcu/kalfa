from torch import nn

from kalfa.std.common.deferred import later


def torch_linear_layer(in_features, out_features, bias):
    return nn.Linear(int(in_features), int(out_features), bias=bool(bias))


def torch_linear(in_features, out_features, bias=True):
    return later(torch_linear_layer, in_features=in_features, out_features=out_features, bias=bias)


def bilinear(in1_features, in2_features, out_features, bias=True):
    return nn.Bilinear(int(in1_features), int(in2_features), int(out_features), bias=bool(bias))


def identity():
    return nn.Identity()
