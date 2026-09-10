import torch

from kalfa.registration import lego


def cuda_device(index):
    if not torch.cuda.is_available():
        raise RuntimeError("cuda is not available on this machine; write device: cpu, mps or auto")
    count = torch.cuda.device_count()
    if not 0 <= int(index) < count:
        raise RuntimeError(f"cuda device index {index} does not exist; this machine has {count} cuda device(s)")
    return torch.device(f"cuda:{int(index)}")


def mps_device():
    if not torch.backends.mps.is_available():
        raise RuntimeError("mps is not available on this machine; write device: cpu, cuda or auto")
    return torch.device("mps")


@lego("/device/kalfa/auto", alias="auto", description="The first available device of cuda, mps, cpu")
def auto():
    for candidate in (cuda_device, mps_device):
        try:
            return candidate(0) if candidate is cuda_device else candidate()
        except RuntimeError:
            continue
    return torch.device("cpu")


@lego("/device/kalfa/cpu", alias="cpu", description="The CPU")
def cpu():
    return torch.device("cpu")


@lego("/device/kalfa/cuda", alias="cuda", description="The cuda device with the given index; an error when cuda "
                                                  "or that index is not available")
def cuda(index=0):
    return cuda_device(index)


@lego("/device/kalfa/mps", alias="mps", description="The Apple mps device; an error when it is not available")
def mps():
    return mps_device()
