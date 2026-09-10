from kalfa.registration import lego
from kalfa.std.pre.base import ToTensor


@lego("/pre/kalfa/to_tensor_signed", alias="to_tensor_signed",
      description="Image to a float tensor in [-1, 1], channels first")
def to_tensor_signed():
    return ToTensor(signed=True)
