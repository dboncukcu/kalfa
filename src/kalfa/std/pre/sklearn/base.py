import numpy

from kalfa.std.pre.base import Scaler


def as_block(values):
    matrix = numpy.asarray(values, dtype="float64")
    return matrix.reshape(-1, 1) if matrix.ndim == 1 else matrix


class SklearnScaler(Scaler):
    grouped = True
    fits = True
    incremental = True
    scaler = None

    def build(self):
        raise NotImplementedError

    def terms(self):
        raise NotImplementedError

    def fit(self, values):
        self.scaler = self.build().fit(as_block(values))

    def partial_fit(self, values):
        if self.scaler is None:
            self.scaler = self.build()
        self.scaler.partial_fit(as_block(values))

    def apply(self, values, columns=None):
        return self.run(values, columns, forward=True)

    def inverse(self, values, columns=None):
        return self.run(values, columns, forward=False)

    def run(self, values, columns, forward):
        matrix = as_block(values)
        width = int(self.scaler.n_features_in_)
        positions = numpy.arange(width) if columns is None else numpy.asarray(columns, dtype=int)
        if len(positions) != matrix.shape[1]:
            raise ValueError(f"the scaler was fitted on {width} columns and got {matrix.shape[1]}")
        shift, scale = self.terms()
        if forward:
            out = (matrix - shift[positions]) / scale[positions]
        else:
            out = matrix * scale[positions] + shift[positions]
        return numpy.asarray(out).reshape(numpy.asarray(values).shape)


class SklearnTransformer(Scaler):
    fits = True
    transformer = None

    def build(self, values):
        raise NotImplementedError

    def fit(self, values):
        self.transformer = self.build(numpy.asarray(values, dtype="float64"))
        self.transformer.fit(numpy.asarray(values, dtype="float64").reshape(-1, 1))

    def apply(self, values):
        return self.transformer.transform(numpy.asarray(values, dtype="float64").reshape(-1, 1)).reshape(-1)

    def inverse(self, values):
        return self.transformer.inverse_transform(
            numpy.asarray(values, dtype="float64").reshape(-1, 1)).reshape(-1)
