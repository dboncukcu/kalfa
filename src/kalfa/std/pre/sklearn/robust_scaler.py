import numpy

from kalfa.registration import lego
from kalfa.std.pre.sklearn.base import SklearnScaler


class RobustScaler(SklearnScaler):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def build(self):
        from sklearn.preprocessing import RobustScaler as Scaler

        return Scaler(quantile_range=(self.low, self.high))

    def terms(self):
        return (numpy.asarray(self.scaler.center_, dtype="float64"),
                numpy.asarray(self.scaler.scale_, dtype="float64"))


@lego("/pre/sklearn/robust_scaler", state=True, alias="robust_scaler", grouped=True,
      description="Center a column on its median and scale it by the distance between the low and high "
                  "percentiles (sklearn RobustScaler); outliers do not move the statistics")
def robust_scaler(low=25.0, high=75.0):
    return RobustScaler(low, high)
