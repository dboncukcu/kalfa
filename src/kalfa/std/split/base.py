import logging

from kalfa.std.common.log import logger_for
from kalfa.std.common.samples import is_samples
from kalfa.std.common.stream import is_stream


logger = logger_for("data.split")


def count_of(part):
    if is_stream(part):
        return part.rows
    return len(part)


def report_sets(name, parts):
    if logger.isEnabledFor(logging.INFO):
        logger.info(f"{name}: train {count_of(parts['train'])}, valid {count_of(parts['valid'])}, "
                    f"test {count_of(parts['test'])}")
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
