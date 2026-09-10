import numpy

from kalfa.registration import lego
from kalfa.std.pre.base import Preprocessor, Scaler


@lego("/pre/kalfa/abs", alias="abs", description="Absolute value of a column")
class Absolute(Preprocessor):
    def apply(self, values):
        return numpy.abs(numpy.asarray(values))


@lego("/pre/kalfa/log", alias="log",
      description="log1p of a column divided by norm, in the given base")
class Log(Scaler):
    def __init__(self, base=10.0, norm=1.0):
        self.base = base
        self.norm = norm

    def apply(self, values):
        scaled = numpy.asarray(values, dtype="float64") / self.norm
        return numpy.log1p(scaled) / numpy.log(self.base)

    def inverse(self, values):
        return numpy.expm1(numpy.asarray(values, dtype="float64") * numpy.log(self.base)) * self.norm


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
