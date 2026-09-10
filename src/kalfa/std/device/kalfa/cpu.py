import torch

from kalfa.registration import lego


@lego("/device/kalfa/cpu", alias="cpu", description="The CPU")
def cpu():
    return torch.device("cpu")
