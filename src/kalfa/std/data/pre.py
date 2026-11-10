import numpy
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from kalfa.core.errors import BuildError


def one_hot():
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def flat(values):
    return numpy.asarray(values).reshape(-1)


class Abs:
    rescales = False

    def fit(self, values):
        return self

    def transform(self, values):
        return numpy.abs(numpy.asarray(values))

    def inverse_transform(self, values):
        return numpy.asarray(values)


class Log:
    def __init__(self, base=10.0, norm=1.0):
        self.base = float(base)
        self.norm = float(norm)
        if self.base <= 0.0 or self.base == 1.0:
            raise BuildError(f"log: base must be positive and not 1, got {base!r}")
        if self.norm <= 0.0:
            raise BuildError(f"log: norm must be positive, got {norm!r}")

    def fit(self, values):
        return self

    def transform(self, values):
        return numpy.log1p(numpy.asarray(values, dtype="float64") / self.norm) / numpy.log(self.base)

    def inverse_transform(self, values):
        return numpy.expm1(numpy.asarray(values, dtype="float64") * numpy.log(self.base)) * self.norm


class Cast:
    rescales = False

    def __init__(self, dtype="float32"):
        try:
            self.dtype = numpy.dtype(dtype)
        except TypeError:
            raise BuildError(f"cast: unknown dtype {dtype!r}") from None

    def fit(self, values):
        return self

    def transform(self, values):
        return numpy.asarray(values).astype(self.dtype)

    def inverse_transform(self, values):
        return numpy.asarray(values)


class Label:
    rescales = False

    def __init__(self):
        self.encoder = LabelEncoder()

    @property
    def classes_(self):
        return self.encoder.classes_

    def fit(self, values):
        self.encoder.fit(flat(values))
        return self

    def transform(self, values):
        return numpy.asarray(self.encoder.transform(flat(values)), dtype="int64")

    def inverse_transform(self, values):
        return self.encoder.inverse_transform(numpy.asarray(values, dtype="int64").reshape(-1))
