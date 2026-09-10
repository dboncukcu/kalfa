import numpy

from kalfa.registration import lego
from kalfa.std.pre.base import Encoder


@lego("/pre/sklearn/kbins_discretizer", state=True, alias="kbins_discretizer",
      description="Cut a column into bins and write them as one hot columns <field>_bin<n> (encode: ordinal "
                  "for one integer column); strategy quantile, uniform or kmeans (sklearn KBinsDiscretizer)")
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
