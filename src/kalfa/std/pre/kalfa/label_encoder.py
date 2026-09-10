import numpy

from kalfa.registration import lego
from kalfa.std.pre.base import Encoder


class LabelEncoder(Encoder):
    def fit(self, values):
        self.classes = numpy.array(sorted(set(numpy.asarray(values).tolist())), dtype=object)
        self.lookup = {value: position for position, value in enumerate(self.classes.tolist())}

    def apply(self, values):
        try:
            return numpy.array([self.lookup[value] for value in numpy.asarray(values).tolist()], dtype="int64")
        except KeyError as exc:
            raise ValueError(f"label {exc.args[0]!r} was not seen when the encoder was fitted") from None

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


@lego("/pre/kalfa/label_encoder", state=True, alias="label_encoder",
      description="Integer codes of a label column, sorted by label; inverted in reports and predictions, "
                  "class scores decode to labels")
def label_encoder():
    return LabelEncoder()
