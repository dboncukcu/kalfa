import numpy

from kalfa.registration import lego
from kalfa.std.pre.sklearn.base import SklearnScaler


class MaxAbsScaler(SklearnScaler):
    def build(self):
        from sklearn.preprocessing import MaxAbsScaler as Scaler

        return Scaler()

    def terms(self):
        scale = numpy.asarray(self.scaler.scale_, dtype="float64")
        return numpy.zeros_like(scale), scale


@lego("/pre/sklearn/max_abs_scaler", state=True, alias="max_abs_scaler", grouped=True,
      description="Scale a column by its largest absolute value, into [-1, 1] with the sign and the zeros "
                  "kept (sklearn MaxAbsScaler); one object over every column that names it")
def max_abs_scaler():
    return MaxAbsScaler()
