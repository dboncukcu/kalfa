import numpy
import pandas

from kalfa.registration import lego
from kalfa.std.pre.base import Preprocessor


@lego("/pre/kalfa/fill", alias="fill",
      description="Fill the missing values of a column without a fit: a constant (a number, or a name such as "
                  "missing that becomes its own category), or method ffill or bfill along the rows")
class Fill(Preprocessor):
    def __init__(self, value=None, method=None):
        if method not in (None, "ffill", "bfill"):
            raise ValueError(f"fill.method must be ffill or bfill, got {method!r}")
        if value is None and method is None:
            raise ValueError("fill needs value or method")
        self.value = value
        self.method = method

    def apply(self, values):
        series = pandas.Series(numpy.asarray(values))
        if self.method == "ffill":
            series = series.ffill()
        elif self.method == "bfill":
            series = series.bfill()
        if self.value is not None:
            series = series.fillna(self.value)
        return series.to_numpy()
