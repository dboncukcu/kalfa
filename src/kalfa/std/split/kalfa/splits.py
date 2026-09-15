import json
import logging
from pathlib import Path

import numpy
import pandas

from kalfa.std.common.log import logger_for
from kalfa.std.common.samples import Samples, is_samples
from kalfa.std.common.stream import is_stream, like
from kalfa.std.source.kalfa.prepared import manifest_of


logger = logger_for("data.split")


def count_of(part):
    if is_stream(part):
        return part.rows
    return len(part)


def report_sets(name, parts):
    if logger.isEnabledFor(logging.INFO):
        logger.info(f"{name}: " + ", ".join(f"{set_name} {count_of(part)}" for set_name, part in parts.items()))
    return parts


def take_rows(df, positions):
    if is_samples(df):
        return df.subset(positions)
    return df.iloc[positions]


def cuts_of(count, ratios):
    if len(ratios) != 3:
        raise ValueError(f"ratios must have three entries (train, valid, test), got {ratios!r}")
    if any(part < 0 for part in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must be non negative and sum to 1, got {ratios!r}")
    first = int(round(count * ratios[0]))
    second = first + int(round(count * ratios[1]))
    return first, min(second, count)


def fold_bounds(count, k):
    sizes = [count // k + (1 if position < count % k else 0) for position in range(k)]
    bounds = [0]
    for size in sizes:
        bounds.append(bounds[-1] + size)
    return bounds


def kfold_counts(count, k, fold, val):
    if not isinstance(k, int) or isinstance(k, bool) or k < 2:
        raise ValueError(f"kfold needs an integer k >= 2, got {k!r}")
    if not isinstance(fold, int) or isinstance(fold, bool) or not 0 <= fold < k:
        raise ValueError(f"fold must be an integer in [0, {k - 1}], got {fold!r}")
    bounds = fold_bounds(count, k)
    held = bounds[fold + 1] - bounds[fold]
    rest = count - held
    carve = int(round(rest * float(val or 0.0)))
    return held, carve, rest - carve


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


def random_split(df, ratios, seed=None):
    if is_stream(df):
        raise ValueError("a stream source cannot be shuffled; the lazy set splits with sequential or given")
    generator = numpy.random.default_rng(seed)
    order = generator.permutation(len(df))
    first, second = cuts_of(len(df), ratios)
    return report_sets("random", {"train": take_rows(df, order[:first]), "valid": take_rows(df, order[first:second]),
                            "test": take_rows(df, order[second:])})


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


def given(df, valid=None, test=None):
    empty = df.empty() if is_stream(df) else df.subset([]) if is_samples(df) else df.iloc[:0]
    return report_sets("given", {"train": df, "valid": read_like(df, valid) if valid is not None else empty,
                           "test": read_like(df, test) if test is not None else empty})


def prepared_split(df, path):
    manifest = manifest_of(path)
    folder = Path(path)
    if is_samples(df):
        parts = {name: df.subset(json.loads((folder / f"{name}.json").read_text())) for name in manifest["sets"]}
    else:
        parts = {name: df[df["kalfa_set"] == name].drop(columns=["kalfa_set"]) for name in manifest["sets"]}
    return report_sets("prepared", parts)
