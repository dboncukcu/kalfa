import numpy
import pandas

from kalfa.std.pre.base import Preprocessor


def most_frequent(values):
    found, counts = numpy.unique(values, return_counts=True)
    return found[int(numpy.argmax(counts))]


class SimpleImputer(Preprocessor):
    fits = True
    strategies = ("mean", "median", "most_frequent", "constant")

    def __init__(self, strategy="mean", fill_value=None, indicator=False):
        if strategy not in self.strategies:
            raise ValueError(f"simple_imputer.strategy must be one of {list(self.strategies)}, got {strategy!r}")
        if strategy == "constant" and fill_value is None:
            raise ValueError("simple_imputer.strategy constant needs fill_value")
        self.strategy = strategy
        self.fill_value = fill_value
        self.indicator = bool(indicator)
        self.statistic = fill_value

    def fit(self, values):
        if self.strategy == "constant":
            return
        column = numpy.asarray(values)
        present = column[~pandas.isna(column)]
        if not len(present):
            self.statistic = 0.0
        elif self.strategy == "mean":
            self.statistic = float(present.astype("float64").mean())
        elif self.strategy == "median":
            self.statistic = float(numpy.median(present.astype("float64")))
        else:
            self.statistic = most_frequent(present)

    def apply(self, values):
        column = numpy.asarray(values)
        mask = pandas.isna(column)
        if not mask.any():
            return column
        out = numpy.array(column, dtype="float64" if column.dtype.kind in "fiu" else object)
        out[mask] = self.statistic
        return out

    def extras(self, values):
        if not self.indicator:
            return {}
        return {"missing": pandas.isna(numpy.asarray(values))}


class Fill(Preprocessor):
    def __init__(self, value=None, method=None):
        if method not in (None, "ffill", "bfill"):
            raise ValueError(f"fill.method must be ffill or bfill, got {method!r}")
        if value is None and method is None:
            raise ValueError("fill needs value or method")
        self.value = value
        self.method = method

    def apply(self, values):
        series = pandas.Series(numpy.asarray(values))
        if self.method == "ffill":
            series = series.ffill()
        elif self.method == "bfill":
            series = series.bfill()
        if self.value is not None:
            series = series.fillna(self.value)
        return series.to_numpy()
