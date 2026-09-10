import numpy

from kalfa.registration import lego
from kalfa.std.pre.base import Preprocessor


class Cast(Preprocessor):
    def __init__(self, dtype):
        self.dtype = dtype

    def apply(self, values):
        return numpy.asarray(values).astype(self.dtype)


@lego("/pre/kalfa/cast", alias="cast", description="Cast a column to a numpy dtype")
def cast(dtype):
    return Cast(dtype)
