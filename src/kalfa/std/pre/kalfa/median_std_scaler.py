import numpy

from kalfa.std.pre.base import Scaler


def as_block(values):
    matrix = numpy.asarray(values, dtype="float64")
    return matrix.reshape(-1, 1) if matrix.ndim == 1 else matrix


class MedianStdScaler(Scaler):
    grouped = True
    fits = True

    def __init__(self):
        self.center = None
        self.scale = None

    def fit(self, values):
        block = as_block(values)
        self.center = numpy.nanmedian(block, axis=0)
        spread = numpy.nanstd(block, axis=0)
        self.scale = numpy.where(spread > 0, spread, 1.0)

    def run(self, values, columns, forward):
        matrix = as_block(values)
        positions = numpy.arange(len(self.center)) if columns is None else numpy.asarray(columns, dtype=int)
        if len(positions) != matrix.shape[1]:
            raise ValueError(f"the scaler was fitted on {len(self.center)} columns and got {matrix.shape[1]}")
        if forward:
            out = (matrix - self.center[positions]) / self.scale[positions]
        else:
            out = matrix * self.scale[positions] + self.center[positions]
        return numpy.asarray(out).reshape(numpy.asarray(values).shape)

    def apply(self, values, columns=None):
        return self.run(values, columns, forward=True)

    def inverse(self, values, columns=None):
        return self.run(values, columns, forward=False)
