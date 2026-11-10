import inspect
import numbers

import torch


TENSORS = ("weight", "pos_weight")


def is_loss(target):
    return inspect.isclass(target) and issubclass(target, torch.nn.modules.loss._Loss)


def numeric(value):
    if isinstance(value, bool):
        return False
    if isinstance(value, numbers.Number):
        return True
    return isinstance(value, list) and bool(value) and all(numeric(item) for item in value)


def coerced(target, kwargs):
    try:
        parameters = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return kwargs
    found = dict(kwargs)
    for name, value in kwargs.items():
        parameter = parameters.get(name)
        if parameter is None:
            continue
        if isinstance(value, list) and isinstance(parameter.default, tuple):
            found[name] = tuple(value)
        elif name in TENSORS and is_loss(target) and numeric(value):
            found[name] = torch.as_tensor(value, dtype=torch.float32)
    return found
