from kalfa.std.source.kalfa.prepared import manifest_of
from kalfa.std.split.kalfa.splits import cuts_of, kfold_counts


def rows_of(path, header):
    if path is None:
        return 0
    if header is None:
        return None
    try:
        return header(path)["rows"]
    except Exception:
        return None


def ratio_sizes(rows, ratios, seed=None, group=None):
    if rows is None:
        cuts_of(1, ratios)
        return {name: (None if ratio > 0 else 0) for name, ratio in zip(("train", "valid", "test"), ratios)}
    first, second = cuts_of(rows, ratios)
    return {"train": first, "valid": second - first, "test": rows - second}


def kfold_sizes(rows, k, fold, val=None, seed=None):
    if rows is None:
        kfold_counts(1, k, fold, val)
        return {"train": None, "valid": None if val else 0, "test": None}
    held, carve, train = kfold_counts(rows, k, fold, val)
    return {"train": train, "valid": carve, "test": held}


def given_sizes(rows, valid=None, test=None, header=None):
    return {"train": rows, "valid": rows_of(valid, header), "test": rows_of(test, header)}


def prepared_sizes(rows, path):
    return dict(manifest_of(path)["sizes"])
