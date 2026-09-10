from kalfa.registration import lego
from kalfa.std.common.samples import is_samples
from kalfa.std.common.stream import is_stream
from kalfa.std.split.base import cuts_of, report_sets
import pandas


@lego("/split/kalfa/sequential", returns=["train", "valid", "test"], refs={"group": "column"},
      alias="sequential", sizes="/lego/kalfa/ratio_sizes",
      description="Cut the rows in their order by ratios; with a group column every group is cut on its own")
def sequential(df, ratios, group=None):
    if is_stream(df):
        if group is not None:
            raise ValueError("a stream source has no group column; drop group for a sequential split")
        first, second = cuts_of(df.rows, ratios)
        return report_sets("sequential", {"train": df.window(0, first), "valid": df.window(first, second),
                                    "test": df.window(second, df.rows)})
    if is_samples(df):
        if group is not None:
            raise ValueError("a Dataset source has no group column; drop group for a sequential split")
        first, second = cuts_of(len(df), ratios)
        positions = numpy.arange(len(df))
        return report_sets("sequential", {"train": df.subset(positions[:first]),
                                    "valid": df.subset(positions[first:second]),
                                    "test": df.subset(positions[second:])})
    parts = {"train": [], "valid": [], "test": []}
    groups = [(None, df)] if group is None else list(df.groupby(group, sort=False))
    for _, part in groups:
        first, second = cuts_of(len(part), ratios)
        parts["train"].append(part.iloc[:first])
        parts["valid"].append(part.iloc[first:second])
        parts["test"].append(part.iloc[second:])
    return report_sets("sequential",
                 {name: pandas.concat(pieces) if pieces else df.iloc[:0] for name, pieces in parts.items()})
