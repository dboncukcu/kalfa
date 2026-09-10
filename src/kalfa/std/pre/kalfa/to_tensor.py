from kalfa.registration import lego
from kalfa.std.pre.base import ToTensor


@lego("/pre/kalfa/to_tensor", alias="to_tensor",
      description="Image to a float tensor in [0, 1], channels first")
def to_tensor():
    return ToTensor()
