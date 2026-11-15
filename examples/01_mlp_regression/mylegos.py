import torch


def log_cosh(pred, target):
    return torch.log(torch.cosh(pred - target)).mean()
