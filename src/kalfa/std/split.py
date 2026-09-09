"""Splits: legos that cut the frame into the train, valid and test sets."""

import logging

from ..registration import lego
from .log import logger_for
from .samples import is_samples
from .stream import is_stream

logger = logger_for("data.split")


def _count(part):
    rows = getattr(part, "rows", None)
    if isinstance(rows, int):
        return rows
    try:
        return len(part)
    except TypeError:
        return "?"


def _sets(name, parts):
    if logger.isEnabledFor(logging.INFO):
        logger.info(f"{name}: train {_count(parts['train'])}, valid {_count(parts['valid'])}, "
                    f"test {_count(parts['test'])}")
    return parts


def _take(df, positions):
    """Rows by position, for a frame or a Dataset source."""
    if is_samples(df):
        return df.subset(positions)
    return df.iloc[positions]


def _cuts(count, ratios):
    if len(ratios) != 3:
        raise ValueError(f"ratios must have three entries (train, valid, test), got {ratios!r}")
    if any(part < 0 for part in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must be non negative and sum to 1, got {ratios!r}")
    first = int(round(count * ratios[0]))
    second = first + int(round(count * ratios[1]))
    return first, min(second, count)


@lego("/split/kalfa/random", returns=["train", "valid", "test"], alias="random_split",
            description="Shuffle the rows with a seed and cut them by ratios into train, valid and test; the short "
                        "form of a split without a uri")
def random(df, ratios, seed=None):
    import numpy

    if is_stream(df):
        raise ValueError("a stream source cannot be shuffled; the lazy set splits with sequential or given")
    generator = numpy.random.default_rng(seed)
    order = generator.permutation(len(df))
    first, second = _cuts(len(df), ratios)
    return _sets("random", {"train": _take(df, order[:first]), "valid": _take(df, order[first:second]),
                            "test": _take(df, order[second:])})


def sizes(rows, ratios):
    """The set sizes a random split of ``rows`` rows produces, for the check's set table."""
    first, second = _cuts(rows, ratios)
    return {"train": first, "valid": second - first, "test": rows - second}


@lego("/split/kalfa/sequential", returns=["train", "valid", "test"], refs={"group": "column"},
            alias="sequential",
            description="Cut the rows in their order by ratios; with a group column every group is cut on its own")
def sequential(df, ratios, group=None):
    import pandas

    if is_stream(df):
        if group is not None:
            raise ValueError("a stream source has no group column; drop group for a sequential split")
        first, second = _cuts(df.rows, ratios)
        return _sets("sequential", {"train": df.window(0, first), "valid": df.window(first, second),
                                    "test": df.window(second, df.rows)})
    if is_samples(df):
        if group is not None:
            raise ValueError("a Dataset source has no group column; drop group for a sequential split")
        first, second = _cuts(len(df), ratios)
        positions = numpy_arange(len(df))
        return _sets("sequential", {"train": df.subset(positions[:first]),
                                    "valid": df.subset(positions[first:second]),
                                    "test": df.subset(positions[second:])})
    parts = {"train": [], "valid": [], "test": []}
    groups = [(None, df)] if group is None else list(df.groupby(group, sort=False))
    for _, part in groups:
        first, second = _cuts(len(part), ratios)
        parts["train"].append(part.iloc[:first])
        parts["valid"].append(part.iloc[first:second])
        parts["test"].append(part.iloc[second:])
    return _sets("sequential",
                 {name: pandas.concat(pieces) if pieces else df.iloc[:0] for name, pieces in parts.items()})


def _fold_bounds(count, k):
    sizes = [count // k + (1 if position < count % k else 0) for position in range(k)]
    bounds = [0]
    for size in sizes:
        bounds.append(bounds[-1] + size)
    return bounds


def _kfold_counts(count, k, fold, val):
    if not isinstance(k, int) or isinstance(k, bool) or k < 2:
        raise ValueError(f"kfold needs an integer k >= 2, got {k!r}")
    if not isinstance(fold, int) or isinstance(fold, bool) or not 0 <= fold < k:
        raise ValueError(f"fold must be an integer in [0, {k - 1}], got {fold!r}")
    bounds = _fold_bounds(count, k)
    held = bounds[fold + 1] - bounds[fold]
    rest = count - held
    carve = int(round(rest * float(val or 0.0)))
    return held, carve, rest - carve


@lego("/split/kalfa/kfold", returns=["train", "valid", "test"], alias="kfold",
            description="k folds of a seeded permutation: the held out fold is the test set, val carves the valid "
                        "set from the rest; without val there is no valid set")
def kfold(df, k, fold, val=None, seed=None):
    import numpy

    if is_stream(df):
        raise ValueError("a stream source cannot be folded; the lazy set splits with sequential or given")
    held, carve, _ = _kfold_counts(len(df), k, fold, val)
    generator = numpy.random.default_rng(seed)
    order = generator.permutation(len(df))
    bounds = _fold_bounds(len(df), k)
    test = order[bounds[fold]:bounds[fold + 1]]
    rest = numpy.concatenate([order[:bounds[fold]], order[bounds[fold + 1]:]])
    return _sets(f"kfold {fold} of {k}", {"train": _take(df, rest[carve:]), "valid": _take(df, rest[:carve]),
                                          "test": _take(df, test)})


def kfold_sizes(rows, params):
    held, carve, train = _kfold_counts(rows, params.get("k"), params.get("fold"), params.get("val"))
    return {"train": train, "valid": carve, "test": held}


def numpy_arange(count):
    import numpy

    return numpy.arange(count)


def _read_like(df, path):
    """A set read from a path the way the train data was read: a Dataset source of the same class, or a table by
    its suffix (parquet, csv)."""
    if is_stream(df):
        from .stream import like

        return like(df, path)
    if is_samples(df):
        from .samples import Samples

        return Samples(type(df.source)(path))
    import pandas

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
    return _sets("given", {"train": df, "valid": _read_like(df, valid) if valid is not None else empty,
                           "test": _read_like(df, test) if test is not None else empty})
