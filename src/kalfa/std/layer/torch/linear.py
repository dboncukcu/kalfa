from torch import nn

from kalfa.registration import lego
from kalfa.std.common.deferred import later


def torch_linear_layer(in_features, out_features):
    return nn.Linear(int(in_features), int(out_features))


@lego("/layer/torch/linear", description="torch.nn.Linear")
def torch_linear(in_features, out_features):
    return later(torch_linear_layer, in_features=in_features, out_features=out_features)
