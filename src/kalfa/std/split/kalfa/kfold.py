from kalfa.registration import lego
from kalfa.std.common.stream import is_stream
from kalfa.std.split.base import fold_bounds, kfold_counts, report_sets, take_rows
import numpy


@lego("/split/kalfa/kfold", returns=["train", "valid", "test"], alias="kfold",
      description="k folds of a seeded permutation: the held out fold is the test set, val carves the valid "
                  "set from the rest; without val there is no valid set")
def kfold(df, k, fold, val=None, seed=None):
    if is_stream(df):
        raise ValueError("a stream source cannot be folded; the lazy set splits with sequential or given")
    held, carve, _ = kfold_counts(len(df), k, fold, val)
    generator = numpy.random.default_rng(seed)
    order = generator.permutation(len(df))
    bounds = fold_bounds(len(df), k)
    test = order[bounds[fold]:bounds[fold + 1]]
    rest = numpy.concatenate([order[:bounds[fold]], order[bounds[fold + 1]:]])
    parts = {"train": take_rows(df, rest[carve:]), "valid": take_rows(df, rest[:carve]), "test": take_rows(df, test)}
    return report_sets(f"kfold {fold} of {k}", parts)
