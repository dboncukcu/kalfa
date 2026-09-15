import numpy

from kalfa.std.pre.sklearn.base import SklearnScaler


class StandardScaler(SklearnScaler):
    def build(self):
        from sklearn import preprocessing

        return preprocessing.StandardScaler()

    def terms(self):
        return numpy.asarray(self.scaler.mean_, dtype="float64"), numpy.asarray(self.scaler.scale_, dtype="float64")


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


class MaxAbsScaler(SklearnScaler):
    def build(self):
        from sklearn import preprocessing

        return preprocessing.MaxAbsScaler()

    def terms(self):
        scale = numpy.asarray(self.scaler.scale_, dtype="float64")
        return numpy.zeros_like(scale), scale


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
