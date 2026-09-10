import numpy

from kalfa.registration import lego
from kalfa.std.pre.base import Scaler


@lego("/pre/kalfa/tanh", alias="tanh",
      description="tanh(x / scale) into (-1, 1); the inverse clips at 1 - eps, so a value that saturated in "
                  "float64 (past about 19 scale) comes back at the clip instead of infinity")
class Tanh(Scaler):
    def __init__(self, scale=1.0, eps=1e-15):
        if scale <= 0.0:
            raise ValueError(f"tanh: scale must be positive, got {scale!r}")
        self.scale = float(scale)
        self.eps = float(eps)

    def apply(self, values):
        return numpy.tanh(numpy.asarray(values, dtype="float64") / self.scale)

    def inverse(self, values):
        limit = 1.0 - self.eps
        clipped = numpy.clip(numpy.asarray(values, dtype="float64"), -limit, limit)
        return numpy.arctanh(clipped) * self.scale
