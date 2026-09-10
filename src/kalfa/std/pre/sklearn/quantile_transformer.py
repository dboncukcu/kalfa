from kalfa.registration import lego
from kalfa.std.pre.sklearn.base import SklearnTransformer


@lego("/pre/sklearn/quantile_transformer", state=True, alias="quantile_transformer",
      description="Map a column onto its own quantiles, uniform or normal (sklearn QuantileTransformer); "
                  "flattens any shape, the inverse interpolates between the stored quantiles")
class QuantileTransformer(SklearnTransformer):
    def __init__(self, quantiles=1000, output="uniform", seed=None):
        self.quantiles = int(quantiles)
        self.output = output
        self.seed = seed

    def build(self, values):
        from sklearn import preprocessing

        return preprocessing.QuantileTransformer(n_quantiles=max(2, min(self.quantiles, len(values))),
                           output_distribution=self.output, random_state=self.seed)
