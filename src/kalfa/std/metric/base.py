import torch


class Metric:
    def reset(self) -> None:
        pass

    def update(self, predictions, targets) -> None:
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError


def as_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach())
    return float(value)
