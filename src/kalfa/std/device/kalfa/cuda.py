from kalfa.registration import lego
from kalfa.std.device.base import cuda_device


@lego("/device/kalfa/cuda", alias="cuda", description="The cuda device with the given index; an error when cuda "
                                                  "or that index is not available")
def cuda(index=0):
    return cuda_device(index)
