import numpy

from kalfa.registration import lego
from kalfa.std.pre.base import Scaler


@lego("/pre/kalfa/atanh", alias="atanh",
      description="artanh(x / scale) of a bounded column, inverted by scale tanh(y); a value outside "
                  "(-scale, scale) is an error that names how many and how large")
class Atanh(Scaler):
    def __init__(self, scale=1.0):
        if scale <= 0.0:
            raise ValueError(f"atanh: scale must be positive, got {scale!r}")
        self.scale = float(scale)

    def apply(self, values):
        scaled = numpy.asarray(values, dtype="float64") / self.scale
        outside = numpy.abs(scaled) >= 1.0
        if outside.any():
            raise ValueError(f"atanh takes values inside (-scale, scale); {int(outside.sum())} of {scaled.size} "
                             f"are outside, the largest is {float(numpy.abs(scaled).max()) * self.scale}; "
                             f"raise scale")
        return numpy.arctanh(scaled)

    def inverse(self, values):
        return numpy.tanh(numpy.asarray(values, dtype="float64")) * self.scale
