from torch import nn

from kalfa.registration import lego
from kalfa.std.common.deferred import later


def torch_linear_layer(in_features, out_features, bias):
    return nn.Linear(int(in_features), int(out_features), bias=bool(bias))


@lego("/layer/torch/linear", description="torch.nn.Linear with in_features written out")
def torch_linear(in_features, out_features, bias=True):
    return later(torch_linear_layer, in_features=in_features, out_features=out_features, bias=bias)


@lego("/layer/torch/bilinear", alias="bilinear",
      description="torch.nn.Bilinear over two inputs of in1_features and in2_features")
def bilinear(in1_features, in2_features, out_features, bias=True):
    return nn.Bilinear(int(in1_features), int(in2_features), int(out_features), bias=bool(bias))


@lego("/layer/torch/identity", alias="identity", description="torch.nn.Identity, the input as it is")
def identity():
    return nn.Identity()
