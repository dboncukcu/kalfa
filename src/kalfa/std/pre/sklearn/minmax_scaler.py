import numpy

from kalfa.registration import lego
from kalfa.std.pre.sklearn.base import SklearnScaler


@lego("/pre/sklearn/minmax_scaler", state=True, alias="minmax_scaler", grouped=True,
      description="Scale a column into [low, high] (sklearn MinMaxScaler); one object over every column that "
                  "names it, its statistics per column")
class MinMaxScaler(SklearnScaler):
    def __init__(self, low=0.0, high=1.0):
        self.low = low
        self.high = high

    def build(self):
        from sklearn import preprocessing

        return preprocessing.MinMaxScaler(feature_range=(self.low, self.high))

    def terms(self):
        scale = numpy.asarray(self.scaler.scale_, dtype="float64")
        return -numpy.asarray(self.scaler.min_, dtype="float64") / scale, 1.0 / scale
