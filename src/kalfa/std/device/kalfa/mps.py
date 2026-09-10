from kalfa.registration import lego
from kalfa.std.device.base import mps_device


@lego("/device/kalfa/mps", alias="mps", description="The Apple mps device; an error when it is not available")
def mps():
    return mps_device()
