import torch

from kalfa.registration import lego
from kalfa.std.device.base import cuda_device, mps_device


@lego("/device/kalfa/auto", alias="auto", description="The first available device of cuda, mps, cpu")
def auto():
    for candidate in (cuda_device, mps_device):
        try:
            return candidate(0) if candidate is cuda_device else candidate()
        except RuntimeError:
            continue
    return torch.device("cpu")
