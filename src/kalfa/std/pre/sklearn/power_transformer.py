from kalfa.registration import lego
from kalfa.std.pre.sklearn.base import SklearnTransformer


@lego("/pre/sklearn/power_transformer", state=True, alias="power_transformer",
      description="Yeo-Johnson (or Box-Cox for positive columns) with the exponent fitted per column, then "
                  "standardized (sklearn PowerTransformer); the invertible way to a near normal column")
class PowerTransformer(SklearnTransformer):
    def __init__(self, method="yeo-johnson", standardize=True):
        self.method = method
        self.standardize = bool(standardize)

    def build(self, values):
        from sklearn import preprocessing

        return preprocessing.PowerTransformer(method=self.method, standardize=self.standardize)
