"""Model level legos: the EMA clone."""

import copy

import torch

from ..registration import lego


def ready(model):
    """Whether a model has real parameters: built, and none of them lazy and uninitialized."""
    if not getattr(model, "initialized", True):
        return False
    return not any(isinstance(parameter, torch.nn.parameter.UninitializedParameter)
                   for parameter in model.parameters())


def freeze(module):
    for parameter in module.parameters():
        if not isinstance(parameter, torch.nn.parameter.UninitializedParameter):
            parameter.requires_grad_(False)
    module.eval()


def frozen_copy(model):
    copied = copy.deepcopy(model)
    if hasattr(copied, "kalfa_trainable"):
        copied.kalfa_trainable = False
    freeze(copied)
    return copied


class Ema(torch.nn.Module):
    """An exponential moving average copy of a model, shifted after every real optimizer step.

    The copy is taken as soon as the model has real parameters: at build time, or after the first batch of a lazy
    model; until then the EMA has no weights.
    """

    def __init__(self, model, decay):
        super().__init__()
        self.decay = float(decay)
        object.__setattr__(self, "source", model)
        self.inputs = list(getattr(model, "inputs", []))
        self.outputs = list(getattr(model, "outputs", []))
        self.model = None
        if ready(model):
            self.model = frozen_copy(model)
        super().train(False)

    def shift(self, model):
        if self.model is None:
            if ready(model):
                self.model = frozen_copy(model)
            return
        with torch.no_grad():
            own = dict(self.model.named_parameters())
            for name, parameter in model.named_parameters():
                own[name].mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)
            buffers = dict(self.model.named_buffers())
            for name, buffer in model.named_buffers():
                buffers[name].copy_(buffer)

    def train(self, mode=True):
        return super().train(False)

    def forward(self, *args):
        if self.model is None:
            raise ValueError("the EMA copy has no weights yet: its model has not taken a batch")
        return self.model(*args)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        inner = {key[len("model."):]: value for key, value in state_dict.items() if key.startswith("model.")}
        if not inner:
            return None
        if self.model is None:
            self.model = frozen_copy(self.source)
        result = self.model.load_state_dict(inner, strict=strict)
        freeze(self.model)
        return result


@lego("/lego/kalfa/clone", state=True,
            description="An exponential moving average copy of a model with the given decay")
def clone(model, decay):
    return Ema(model, decay)
