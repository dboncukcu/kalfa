import numpy

from kalfa.registration import lego
from kalfa.std.pre.base import Preprocessor


class Absolute(Preprocessor):
    def apply(self, values):
        return numpy.abs(numpy.asarray(values))


@lego("/pre/kalfa/abs", alias="abs", description="Absolute value of a column")
def absolute():
    return Absolute()
