from kalfa.registration import lego
from kalfa.std.split.base import kfold_counts


@lego("/lego/kalfa/kfold_sizes",
      description="The set sizes a k fold split produces from rows rows; without rows, which sets it produces")
def kfold_sizes(rows, k, fold, val=None, seed=None):
    if rows is None:
        kfold_counts(1, k, fold, val)
        return {"train": None, "valid": None if val else 0, "test": None}
    held, carve, train = kfold_counts(rows, k, fold, val)
    return {"train": train, "valid": carve, "test": held}
