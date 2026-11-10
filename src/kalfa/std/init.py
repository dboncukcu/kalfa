from torch import nn
from torch.nn.parameter import is_lazy

from kalfa.core.aliases import register


LAYERS = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
          nn.Embedding)


def zero_bias(layer):
    bias = getattr(layer, "bias", None)
    if bias is not None:
        nn.init.zeros_(bias)


class Pending:
    def __init__(self, fill):
        self.fill = fill
        self.done = False

    def __call__(self, module, inputs):
        if not self.done:
            self.done = True
            self.fill(module)


def initializer(fill):
    def apply(module):
        for layer in module.modules():
            if not isinstance(layer, LAYERS):
                continue
            if any(is_lazy(parameter) for parameter in layer.parameters(recurse=False)):
                layer.register_forward_pre_hook(Pending(fill))
            else:
                fill(layer)

    return apply


def normal_init(std=0.02, mean=0.0):
    def fill(layer):
        nn.init.normal_(layer.weight, mean=float(mean), std=float(std))
        zero_bias(layer)

    return initializer(fill)


def xavier_init(gain=1.0):
    def fill(layer):
        nn.init.xavier_uniform_(layer.weight, gain=float(gain))
        zero_bias(layer)

    return initializer(fill)


INITS = (
    ("normal_init", "kalfa.std.init.normal_init", "draws linear, convolution and embedding weights from "
                                                  "normal(mean, std) and zeroes biases; lazy layers are filled on "
                                                  "their first forward"),
    ("xavier_init", "kalfa.std.init.xavier_init", "xavier uniform weights for linear, convolution and embedding "
                                                  "layers, zero biases; lazy layers are filled on their first "
                                                  "forward"),
)


def register_names():
    for name, target, doc in INITS:
        register(name, target, slot="value", doc=doc)


register_names()
