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
