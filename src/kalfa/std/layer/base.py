from torch import nn


class LazyLayer(nn.Module):
    pass


def linear_layer(out_features, in_features=None):
    if in_features is None:
        return nn.LazyLinear(int(out_features))
    return nn.Linear(int(in_features), int(out_features))
