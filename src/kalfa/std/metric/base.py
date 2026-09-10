import torch


class Metric:
    def reset(self):
        pass

    def update(self, predictions, targets):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError


def as_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach())
    return float(value)
