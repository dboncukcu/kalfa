import numpy

from kalfa.registration import lego
from kalfa.std.pre.base import Encoder, Preprocessor


@lego("/pre/kalfa/cast", alias="cast", description="Cast a column to a numpy dtype")
class Cast(Preprocessor):
    def __init__(self, dtype):
        self.dtype = dtype

    def apply(self, values):
        return numpy.asarray(values).astype(self.dtype)


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


@lego("/pre/kalfa/label_encoder", state=True, alias="label_encoder",
      description="Integer codes of a label column, sorted by label; inverted in reports and predictions, "
                  "class scores decode to labels")
class LabelEncoder(Encoder):
    decodes = True

    def fit(self, values):
        self.classes = numpy.array(sorted(set(numpy.asarray(values).tolist())), dtype=object)
        self.lookup = {value: position for position, value in enumerate(self.classes.tolist())}

    def apply(self, values):
        try:
            return numpy.array([self.lookup[value] for value in numpy.asarray(values).tolist()], dtype="int64")
        except KeyError as exception:
            raise ValueError(f"label {exception.args[0]!r} was not seen when the encoder was fitted") from None

    def inverse(self, values):
        codes = numpy.asarray(values).astype("int64").reshape(-1)
        return self.classes[codes]

    def decode(self, scores):
        matrix = numpy.asarray(scores)
        if matrix.ndim == 1 or matrix.shape[1] == 1:
            codes = (matrix.reshape(-1) > 0).astype("int64")
        else:
            codes = matrix.argmax(axis=1)
        return self.classes[codes]
