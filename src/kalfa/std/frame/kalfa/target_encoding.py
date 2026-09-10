import numpy

from kalfa.registration import lego
from kalfa.std.frame.base import FrameTransform, table_only


@lego("/frame/kalfa/target_encoding", alias="target_encoding", needs_table=True, refs={"column": "column"},
      description="The train mean of the target per category of a column, smoothed toward the overall mean by "
                  "smoothing pseudo counts, as a new column (name, or <column>_target); a category the train set "
                  "never saw takes the overall mean")
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
