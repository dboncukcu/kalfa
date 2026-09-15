import numpy

from kalfa.std.pre.base import Preprocessor, Scaler


class Absolute(Preprocessor):
    def apply(self, values):
        return numpy.abs(numpy.asarray(values))


class Log(Scaler):
    def __init__(self, base=10.0, norm=1.0):
        self.base = base
        self.norm = norm

    def apply(self, values):
        scaled = numpy.asarray(values, dtype="float64") / self.norm
        return numpy.log1p(scaled) / numpy.log(self.base)

    def inverse(self, values):
        return numpy.expm1(numpy.asarray(values, dtype="float64") * numpy.log(self.base)) * self.norm


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
