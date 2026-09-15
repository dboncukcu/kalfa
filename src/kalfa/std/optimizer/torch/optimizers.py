import torch

from kalfa.std.optimizer.base import Optimizer


class Sgd(Optimizer):
    torch_class = torch.optim.SGD


class Adam(Optimizer):
    torch_class = torch.optim.Adam


class AdamW(Optimizer):
    torch_class = torch.optim.AdamW
