import numpy

from kalfa.registration import lego
from kalfa.std.pre.base import Scaler


@lego("/pre/kalfa/asinh", alias="asinh",
      description="Signed log scale of a heavy tailed column: arcsinh(x / scale), inverted by scale sinh(y); "
                  "keeps the sign, linear near zero, logarithmic in the tails, defined at zero; the inverse "
                  "refuses values past overflow, where sinh leaves float64")
class Asinh(Scaler):
    def __init__(self, scale=1.0, overflow=700.0):
        if scale <= 0.0:
            raise ValueError(f"asinh: scale must be positive, got {scale!r}")
        self.scale = float(scale)
        self.overflow = float(overflow)

    def apply(self, values):
        return numpy.arcsinh(numpy.asarray(values, dtype="float64") / self.scale)

    def inverse(self, values):
        out = numpy.asarray(values, dtype="float64")
        largest = float(numpy.abs(out).max()) if out.size else 0.0
        if largest > self.overflow:
            raise ValueError(f"asinh: the inverse overflows at {largest}, past overflow={self.overflow}")
        return numpy.sinh(out) * self.scale
