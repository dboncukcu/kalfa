import numpy

from kalfa.registration import lego
from kalfa.std.pre.base import Scaler


@lego("/pre/kalfa/sinh", alias="sinh",
      description="sinh(x / scale), the direction opposite to asinh: it stretches the tails instead of "
                  "compressing them; a value past overflow is an error, where sinh leaves float64")
class Sinh(Scaler):
    def __init__(self, scale=1.0, overflow=700.0):
        if scale <= 0.0:
            raise ValueError(f"sinh: scale must be positive, got {scale!r}")
        self.scale = float(scale)
        self.overflow = float(overflow)

    def apply(self, values):
        out = numpy.asarray(values, dtype="float64") / self.scale
        largest = float(numpy.abs(out).max()) if out.size else 0.0
        if largest > self.overflow:
            raise ValueError(f"sinh: overflows at {largest} / scale, past overflow={self.overflow}; raise scale")
        return numpy.sinh(out)

    def inverse(self, values):
        return numpy.arcsinh(numpy.asarray(values, dtype="float64")) * self.scale
