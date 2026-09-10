import numpy

from kalfa.std.pre.base import Scaler


class SklearnScaler(Scaler):
    """A sklearn scaler over every column that names it: one object, per column statistics (``grouped``).

    ``fit`` takes the matrix of the columns in plan order; ``apply`` and ``inverse`` take the whole matrix, or a
    slice of it with ``columns`` naming the positions the slice holds.
    """

    rescales = True
    grouped = True

    def build(self):
        raise NotImplementedError

    def fit(self, values):
        self.scaler = self.build().fit(as_block(values))

    def partial_fit(self, values):
        if getattr(self, "scaler", None) is None:
            self.scaler = self.build()
        self.scaler.partial_fit(as_block(values))

    def apply(self, values, columns=None):
        return self._run(values, columns, forward=True)

    def inverse(self, values, columns=None):
        return self._run(values, columns, forward=False)

    def _run(self, values, columns, forward):
        matrix = as_block(values)
        width = int(getattr(self.scaler, "n_features_in_", matrix.shape[1]))
        positions = numpy.arange(width) if columns is None else numpy.asarray(columns, dtype=int)
        if len(positions) != matrix.shape[1]:
            raise ValueError(f"the scaler was fitted on {width} columns and got {matrix.shape[1]}")
        shift, scale = self.terms()
        out = (matrix - shift[positions]) / scale[positions] if forward else matrix * scale[positions] + shift[positions]
        return numpy.asarray(out).reshape(numpy.asarray(values).shape)

    def terms(self):
        """The per column (shift, scale) of the fitted scaler: apply is (value - shift) / scale."""
        raise NotImplementedError


def as_block(values):
    matrix = numpy.asarray(values, dtype="float64")
    return matrix.reshape(-1, 1) if matrix.ndim == 1 else matrix


class SklearnTransformer(Scaler):
    """A sklearn transformer fitted per column: the work is per feature anyway, so no grouping to win."""

    rescales = True

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
