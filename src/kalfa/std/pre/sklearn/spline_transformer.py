import numpy

from kalfa.registration import lego
from kalfa.std.pre.base import Encoder


class Spline(Encoder):
    def __init__(self, knots, degree, extrapolation):
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


@lego("/pre/sklearn/spline_transformer", state=True, alias="spline_transformer",
      description="A B-spline basis of a column, <field>_spline<n>: a smooth non linear expansion of one "
                  "feature that a linear head can use (sklearn SplineTransformer)")
def spline_transformer(knots=5, degree=3, extrapolation="constant"):
    return Spline(knots, degree, extrapolation)
