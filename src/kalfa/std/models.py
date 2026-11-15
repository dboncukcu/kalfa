import copy

from torch import nn

from kalfa.core import aliases
from kalfa.core.aliases import register


ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "sigmoid": nn.Sigmoid,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
    "softplus": nn.Softplus,
}


def activation_of(activation):
    if isinstance(activation, nn.Module):
        return copy.deepcopy(activation)
    if isinstance(activation, str):
        found = ACTIVATIONS.get(activation)
        if found is None:
            raise ValueError(f"unknown activation {activation!r}; one of {sorted(ACTIVATIONS)}")
        return found()
    if callable(activation):
        return activation()
    raise TypeError(f"activation must be a name, a module or a module class, got {type(activation).__name__}")


def linear_of(width, size):
    return nn.LazyLinear(size) if width is None else nn.Linear(width, size)


class Mlp(nn.Module):
    def __init__(self, hidden, out, activation="relu", dropout=0.0, in_features=None):
        super().__init__()
        sizes = [int(hidden)] if isinstance(hidden, int) else [int(size) for size in hidden]
        width = None if in_features is None else int(in_features)
        layers = []
        for size in sizes:
            layers.append(linear_of(width, size))
            layers.append(activation_of(activation))
            if dropout and float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            width = size
        layers.append(linear_of(width, int(out)))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def layer_of(item):
    if isinstance(item, nn.Module):
        return item
    if isinstance(item, str):
        target = aliases.resolve(item)
        if isinstance(target, str):
            target = aliases.import_target(target)
        return target()
    if callable(item):
        return item()
    raise TypeError(f"a sequential layer is a module, a short name or a module class, got {type(item).__name__}")


class Sequential(nn.Module):
    def __init__(self, layers):
        super().__init__()
        if isinstance(layers, (str, nn.Module)) or not hasattr(layers, "__iter__"):
            raise TypeError("sequential takes layers: a list of modules or short names")
        self.layers = nn.Sequential(*[layer_of(item) for item in layers])

    def forward(self, *inputs):
        value = inputs[0] if len(inputs) == 1 else inputs
        return self.layers(value)


class LastStep(nn.Module):
    def forward(self, value):
        if isinstance(value, (tuple, list)):
            value = value[0]
        return value[:, -1]


MODELS = (
    ("mlp", "kalfa.std.models.Mlp", "linear layers over hidden sizes with an activation, dropout and a final "
                                    "linear to out; the first layer infers its input width when in_features is "
                                    "not given"),
    ("stack", "kalfa.std.models.Sequential", "a stack of layers applied in order: call mappings such as "
                                             "{use: linear, out_features: 8} or short names of argument free layers "
                                             "such as relu, flatten or last_step"),
)

LAYERS = (
    ("linear", "torch.nn.LazyLinear", "a linear layer that infers its input width: {use: linear, out_features: n}"),
    ("relu", "torch.nn.ReLU", "rectified linear activation"),
    ("gelu", "torch.nn.GELU", "gaussian error linear activation"),
    ("tanh", "torch.nn.Tanh", "hyperbolic tangent activation"),
    ("sigmoid", "torch.nn.Sigmoid", "sigmoid activation"),
    ("silu", "torch.nn.SiLU", "sigmoid weighted linear activation"),
    ("dropout", "torch.nn.Dropout", "dropout with probability p"),
    ("flatten", "torch.nn.Flatten", "flattens every dimension after the batch dimension"),
    ("layer_norm", "torch.nn.LayerNorm", "layer normalisation over normalized_shape"),
    ("embedding", "torch.nn.Embedding", "a lookup table: {use: embedding, num_embeddings: n, embedding_dim: d}"),
    ("batch_norm", "torch.nn.LazyBatchNorm1d", "batch normalisation over features, width inferred"),
    ("conv2d", "torch.nn.LazyConv2d", "a 2d convolution that infers its input channels"),
    ("last_step", "kalfa.std.models.LastStep", "the last time step of a batch first sequence output, also from the "
                                               "(output, state) tuple a GRU or LSTM returns"),
)

def register_names():
    for name, target, doc in MODELS:
        register(name, target, slot="model", doc=doc)
    for name, target, doc in LAYERS:
        register(name, target, slot="value", doc=doc)


register_names()
