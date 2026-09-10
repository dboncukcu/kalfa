import numpy

from kalfa.registration import lego
from kalfa.std.pre.base import Scaler


class Log(Scaler):
    rescales = True

    def __init__(self, base, norm):
        self.base = base
        self.norm = norm

    def apply(self, values):
        scaled = numpy.asarray(values, dtype="float64") / self.norm
        return numpy.log1p(scaled) / numpy.log(self.base)

    def inverse(self, values):
        return numpy.expm1(numpy.asarray(values, dtype="float64") * numpy.log(self.base)) * self.norm


@lego("/pre/kalfa/log", alias="log",
      description="log1p of a column divided by norm, in the given base")
def log(base=10.0, norm=1.0):
    return Log(base, norm)
