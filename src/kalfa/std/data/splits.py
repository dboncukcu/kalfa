import numpy
import pandas

from kalfa.core.errors import BuildError


SET_NAMES = ("train", "valid", "test")


def ratios_of(ratios):
    if isinstance(ratios, (str, bytes, dict, bool)):
        raise BuildError(f"split ratios are up to three shares (train, valid, test), got {ratios!r}")
    if isinstance(ratios, (int, float)):
        ratios = [ratios]
    try:
        values = [float(part) for part in ratios]
    except (TypeError, ValueError):
        raise BuildError(f"split ratios must be numbers, got {ratios!r}") from None
    if not 1 <= len(values) <= 3:
        raise BuildError(f"split ratios are up to three shares (train, valid, test), got {len(values)} entries")
    values += [0.0] * (3 - len(values))
    if any(part < 0.0 for part in values) or abs(sum(values) - 1.0) > 1e-6:
        raise BuildError(f"split ratios must be non negative and sum to 1, got {ratios!r}")
    return values


def sizes_of(count, ratios):
    active = [position for position, share in enumerate(ratios) if share > 0.0]
    sizes = [0, 0, 0]
    used = 0
    for position in active[:-1]:
        sizes[position] = int(count * ratios[position] + 1e-9)
        used += sizes[position]
    sizes[active[-1]] = max(count - used, 0)
    return sizes


def pieces_of(order, ratios):
    pieces = {}
    start = 0
    for name, size in zip(SET_NAMES, sizes_of(len(order), ratios)):
        pieces[name] = order[start:start + size]
        start += size
    return pieces


def sets_of(frame, pieces):
    return {name: frame.iloc[numpy.sort(piece)] for name, piece in pieces.items() if len(piece)}


def permutation(count, seed):
    return numpy.random.default_rng(seed).permutation(count)


def fold_bounds(count, folds):
    sizes = [count // folds + (1 if position < count % folds else 0) for position in range(folds)]
    bounds = [0]
    for size in sizes:
        bounds.append(bounds[-1] + size)
    return bounds


def whole(value, what, low, high=None):
    if isinstance(value, bool) or not isinstance(value, int) or value < low or (high is not None and value > high):
        span = f"at least {low}" if high is None else f"in [{low}, {high}]"
        raise BuildError(f"{what} must be an integer {span}, got {value!r}")
    return value


class RandomSplit:
    def __init__(self, ratios, seed=None):
        self.ratios = ratios_of(ratios)
        self.seed = seed

    def split(self, frame):
        return sets_of(frame, pieces_of(permutation(len(frame), self.seed), self.ratios))


class SequentialSplit:
    def __init__(self, ratios, group=None):
        self.ratios = ratios_of(ratios)
        self.group = group

    def split(self, frame):
        if self.group is None:
            return sets_of(frame, pieces_of(numpy.arange(len(frame)), self.ratios))
        if self.group not in frame.columns:
            raise BuildError(f"sequential split: the group column {self.group!r} is not in the table")
        codes, _ = pandas.factorize(frame[self.group], sort=False, use_na_sentinel=False)
        collected = {name: [] for name in SET_NAMES}
        for code in range(int(codes.max()) + 1 if len(codes) else 0):
            members = numpy.flatnonzero(codes == code)
            for name, piece in pieces_of(members, self.ratios).items():
                collected[name].append(piece)
        pieces = {name: numpy.concatenate(parts) if parts else numpy.zeros(0, dtype="int64")
                  for name, parts in collected.items()}
        return sets_of(frame, pieces)


class KFoldSplit:
    def __init__(self, folds, fold, seed=None, valid_ratio=0.0):
        self.folds = whole(folds, "kfold: folds", 2)
        self.fold = whole(fold, "kfold: fold", 0, self.folds - 1)
        self.seed = seed
        self.valid_ratio = float(valid_ratio or 0.0)
        if not 0.0 <= self.valid_ratio < 1.0:
            raise BuildError(f"kfold: valid_ratio is a share of the train rows in [0, 1), got {valid_ratio!r}")

    def split(self, frame):
        order = permutation(len(frame), self.seed)
        bounds = fold_bounds(len(order), self.folds)
        test = order[bounds[self.fold]:bounds[self.fold + 1]]
        rest = numpy.concatenate([order[:bounds[self.fold]], order[bounds[self.fold + 1]:]])
        carve = int(round(len(rest) * self.valid_ratio))
        return sets_of(frame, {"train": rest[carve:], "valid": rest[:carve], "test": test})


class GivenSplit:
    def __init__(self, column, train="train", valid="valid", test="test"):
        self.column = column
        self.train = train
        self.valid = valid
        self.test = test

    def split(self, frame):
        if self.column not in frame.columns:
            raise BuildError(f"given split: the column {self.column!r} is not in the table")
        values = frame[self.column]
        labels = {"train": self.train, "valid": self.valid, "test": self.test}
        out = {}
        for name, label in labels.items():
            if label is None:
                continue
            mask = (values == label).to_numpy()
            if mask.any():
                out[name] = frame[mask]
        if not out:
            wanted = [label for label in labels.values() if label is not None]
            raise BuildError(f"given split: the column {self.column!r} holds none of {wanted}")
        return out
