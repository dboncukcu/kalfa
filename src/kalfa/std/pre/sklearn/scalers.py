import numpy

from kalfa.registration import lego
from kalfa.std.pre.sklearn.base import SklearnScaler


@lego("/pre/sklearn/standard_scaler", state=True, alias="standard_scaler", grouped=True,
      description="Standardize a column to zero mean and unit variance (sklearn StandardScaler); one object "
                  "over every column that names it, its statistics per column")
class StandardScaler(SklearnScaler):
    def build(self):
        from sklearn import preprocessing

        return preprocessing.StandardScaler()

    def terms(self):
        return numpy.asarray(self.scaler.mean_, dtype="float64"), numpy.asarray(self.scaler.scale_, dtype="float64")


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


@lego("/pre/sklearn/max_abs_scaler", state=True, alias="max_abs_scaler", grouped=True,
      description="Scale a column by its largest absolute value, into [-1, 1] with the sign and the zeros "
                  "kept (sklearn MaxAbsScaler); one object over every column that names it")
class MaxAbsScaler(SklearnScaler):
    def build(self):
        from sklearn import preprocessing

        return preprocessing.MaxAbsScaler()

    def terms(self):
        scale = numpy.asarray(self.scaler.scale_, dtype="float64")
        return numpy.zeros_like(scale), scale


@lego("/pre/sklearn/robust_scaler", state=True, alias="robust_scaler", grouped=True,
      description="Center a column on its median and scale it by the distance between the low and high "
                  "percentiles (sklearn RobustScaler); outliers do not move the statistics")
class RobustScaler(SklearnScaler):
    def __init__(self, low=25.0, high=75.0):
        self.low = low
        self.high = high

    def build(self):
        from sklearn import preprocessing

        return preprocessing.RobustScaler(quantile_range=(self.low, self.high))

    def terms(self):
        return (numpy.asarray(self.scaler.center_, dtype="float64"),
                numpy.asarray(self.scaler.scale_, dtype="float64"))
