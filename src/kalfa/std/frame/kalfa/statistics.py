import numpy

from kalfa.std.frame.base import FrameTransform, table_only


class GroupStatistic(FrameTransform):
    statistics = ("mean", "median", "min", "max", "std", "count")

    def __init__(self, by, column, statistic="mean", name=None):
        if statistic not in self.statistics:
            raise ValueError(f"group_statistic.statistic must be one of {list(self.statistics)}, got {statistic!r}")
        self.by = [by] if isinstance(by, str) else list(by)
        self.column = column
        self.statistic = statistic
        self.name = name or f"{column}_{statistic}_by_{'_'.join(self.by)}"
        self.table = None
        self.overall = None

    def fit(self, df):
        table = table_only(df, "group_statistic")
        self.table = table.groupby(self.by)[self.column].agg(self.statistic)
        self.overall = float(table[self.column].agg(self.statistic))

    def apply(self, df):
        table = table_only(df, "group_statistic")
        keys = table[self.by[0]] if len(self.by) == 1 else list(zip(*[table[key] for key in self.by]))
        values = self.table.reindex(keys).to_numpy(dtype="float64")
        return table.assign(**{self.name: numpy.where(numpy.isnan(values), self.overall, values)})


class TargetEncoding(FrameTransform):
    def __init__(self, column, target, smoothing=1.0, name=None):
        self.column = column
        self.target = target
        self.smoothing = float(smoothing)
        self.name = name or f"{column}_target"
        self.table = None
        self.overall = None

    def fit(self, df):
        table = table_only(df, "target_encoding")
        self.overall = float(table[self.target].mean())
        groups = table.groupby(self.column)[self.target].agg(["sum", "count"])
        self.table = (groups["sum"] + self.smoothing * self.overall) / (groups["count"] + self.smoothing)

    def apply(self, df):
        table = table_only(df, "target_encoding")
        values = self.table.reindex(table[self.column]).to_numpy(dtype="float64")
        return table.assign(**{self.name: numpy.where(numpy.isnan(values), self.overall, values)})
