import torch


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
