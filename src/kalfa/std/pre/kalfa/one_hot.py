import numpy

from kalfa.registration import lego
from kalfa.std.pre.base import Encoder


@lego("/pre/kalfa/one_hot", state=True, alias="one_hot",
      description="One hot columns <field>_<category> of a categorical column; unknown categories give zeros")
class OneHot(Encoder):
    def fit(self, values):
        from sklearn.preprocessing import OneHotEncoder

        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(
            numpy.asarray(values, dtype=object).reshape(-1, 1))
        self.categories = [str(category) for category in self.encoder.categories_[0]]

    def apply(self, values):
        return self.encoder.transform(numpy.asarray(values, dtype=object).reshape(-1, 1)).astype("float32")

    def columns(self, name):
        return [f"{name}_{category}" for category in self.categories]
