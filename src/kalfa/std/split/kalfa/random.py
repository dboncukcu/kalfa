from kalfa.registration import lego
from kalfa.std.common.stream import is_stream
from kalfa.std.split.base import cuts_of, report_sets, take_rows
import numpy


@lego("/split/kalfa/random", returns=["train", "valid", "test"], alias="random_split",
      description="Shuffle the rows with a seed and cut them by ratios into train, valid and test; the short "
                  "form of a split without a uri")
def random_split(df, ratios, seed=None):
    if is_stream(df):
        raise ValueError("a stream source cannot be shuffled; the lazy set splits with sequential or given")
    generator = numpy.random.default_rng(seed)
    order = generator.permutation(len(df))
    first, second = cuts_of(len(df), ratios)
    return report_sets("random", {"train": take_rows(df, order[:first]), "valid": take_rows(df, order[first:second]),
                            "test": take_rows(df, order[second:])})
