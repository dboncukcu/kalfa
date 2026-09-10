from kalfa.registration import lego
from kalfa.std.common.samples import is_samples
from kalfa.std.common.stream import is_stream
from kalfa.std.split.base import report_sets
import pandas
from kalfa.std.common.samples import Samples
from kalfa.std.common.stream import like


def read_like(df, path):
    if is_stream(df):
        return like(df, path)
    if is_samples(df):
        return Samples(type(df.source)(path))

    suffix = str(path).rsplit(".", 1)[-1].lower()
    if suffix == "parquet":
        return pandas.read_parquet(path)
    if suffix in ("csv", "txt"):
        return pandas.read_csv(path)
    raise ValueError(f"given: cannot read {path!r}; a table set is a .parquet or .csv file")


@lego("/split/kalfa/given", returns=["train", "valid", "test"], alias="given",
      description="The source is the train set; valid and test come from the given paths, read like the "
                  "source (a missing path means no set)")
def given(df, valid=None, test=None):
    empty = df.empty() if is_stream(df) else df.subset([]) if is_samples(df) else df.iloc[:0]
    return report_sets("given", {"train": df, "valid": read_like(df, valid) if valid is not None else empty,
                           "test": read_like(df, test) if test is not None else empty})
