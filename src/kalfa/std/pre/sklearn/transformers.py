import numpy

from kalfa.std.pre.base import Encoder
from kalfa.std.pre.sklearn.base import SklearnTransformer


class QuantileTransformer(SklearnTransformer):
    def __init__(self, quantiles=1000, output="uniform", seed=None):
        self.quantiles = int(quantiles)
        self.output = output
        self.seed = seed

    def build(self, values):
        from sklearn import preprocessing

        return preprocessing.QuantileTransformer(n_quantiles=max(2, min(self.quantiles, len(values))),
                           output_distribution=self.output, random_state=self.seed)


class PowerTransformer(SklearnTransformer):
    def __init__(self, method="yeo-johnson", standardize=True):
        self.method = method
        self.standardize = bool(standardize)

    def build(self, values):
        from sklearn import preprocessing

        return preprocessing.PowerTransformer(method=self.method, standardize=self.standardize)


class KBins(Encoder):
    def __init__(self, bins=5, strategy="quantile", encode="onehot"):
        self.bins = int(bins)
        self.strategy = strategy
        self.encode = encode

    def fit(self, values):
        from sklearn.preprocessing import KBinsDiscretizer

        kind = "onehot-dense" if self.encode == "onehot" else "ordinal"
        self.encoder = KBinsDiscretizer(n_bins=self.bins, encode=kind, strategy=self.strategy)
        self.encoder.fit(numpy.asarray(values, dtype="float64").reshape(-1, 1))
        self.width = int(self.encoder.n_bins_[0]) if self.encode == "onehot" else 1

    def apply(self, values):
        out = self.encoder.transform(numpy.asarray(values, dtype="float64").reshape(-1, 1))
        return numpy.asarray(out, dtype="float32").reshape(len(numpy.asarray(values)), -1) \
            if self.encode == "onehot" else numpy.asarray(out, dtype="float32").reshape(-1)

    def columns(self, name):
        return [f"{name}_bin{position}" for position in range(self.width)]


class Spline(Encoder):
    def __init__(self, knots=5, degree=3, extrapolation="constant"):
        self.knots = int(knots)
        self.degree = int(degree)
        self.extrapolation = extrapolation

    def fit(self, values):
        from sklearn.preprocessing import SplineTransformer

        self.transformer = SplineTransformer(n_knots=self.knots, degree=self.degree,
                                             extrapolation=self.extrapolation, include_bias=False)
        self.transformer.fit(numpy.asarray(values, dtype="float64").reshape(-1, 1))
        self.width = len(self.transformer.get_feature_names_out())

    def apply(self, values):
        out = self.transformer.transform(numpy.asarray(values, dtype="float64").reshape(-1, 1))
        return numpy.asarray(out, dtype="float32")

    def columns(self, name):
        return [f"{name}_spline{position}" for position in range(self.width)]
