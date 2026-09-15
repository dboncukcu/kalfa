from torch import nn

from kalfa.std.common.deferred import later


def linear_layer(out_features, in_features=None):
    if in_features is None:
        return nn.LazyLinear(int(out_features))
    return nn.Linear(int(in_features), int(out_features))


def activation_layer(name):
    kinds = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh, "leaky_relu": nn.LeakyReLU,
             "elu": nn.ELU, "selu": nn.SELU, "mish": nn.Mish, "sigmoid": nn.Sigmoid}
    if name not in kinds:
        raise ValueError(f"mlp.activation must be one of {sorted(kinds)}, got {name!r}")
    return kinds[name]()


def linear(out_features, in_features=None):
    return later(linear_layer, out_features=out_features, in_features=in_features)


def linear_relu(out_features, in_features=None):
    return nn.Sequential(later(linear_layer, out_features=out_features, in_features=in_features), nn.ReLU())


def mlp(widths, activation="relu", dropout=0.0, out_features=None, in_features=None):
    parts = []
    for position, width in enumerate(widths):
        parts.append(later(linear_layer, out_features=width, in_features=in_features if position == 0 else None))
        parts.append(activation_layer(activation))
        if float(dropout) > 0.0:
            parts.append(nn.Dropout(float(dropout)))
    if out_features is not None:
        parts.append(later(linear_layer, out_features=out_features, in_features=None if widths else in_features))
    return nn.Sequential(*parts)
