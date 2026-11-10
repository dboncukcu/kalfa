import logging
import numbers
from contextlib import contextmanager

import torch
from torch import nn

from kalfa.core.errors import BuildError


logger = logging.getLogger("kalfa.std")


def positive_int(value, what):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise BuildError(f"{what} must be a positive integer, got {value!r}")
    return int(value)


def numeric(value):
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


@contextmanager
def evaluating(model):
    modules = list(model.modules()) if isinstance(model, nn.Module) else []
    modes = [module.training for module in modules]
    if modules:
        model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        for module, mode in zip(modules, modes):
            module.training = mode


def classes_last_of(pred, target, choice, owner, label):
    if choice is not None:
        return bool(choice)
    if pred.ndim < 3 or target.ndim != pred.ndim - 1 or pred.shape[0] != target.shape[0]:
        return None
    last = tuple(pred.shape[1:-1]) == tuple(target.shape[1:])
    first = tuple(pred.shape[2:]) == tuple(target.shape[1:])
    if last and first:
        if not getattr(owner, "ambiguous_warned", False):
            owner.ambiguous_warned = True
            logger.warning(f"{label}: logits of shape {tuple(pred.shape)} fit both (batch, tokens, vocabulary) and "
                           f"(batch, vocabulary, tokens); the vocabulary is taken in dim 1 as torch does; write "
                           f"classes_last: true for (batch, tokens, vocabulary)")
        return False
    if last:
        return True
    if first:
        return False
    return None
