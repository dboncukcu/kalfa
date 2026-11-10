import numpy
import pandas
import torch
from torch.utils.data import Dataset

from kalfa.core.aliases import register
from kalfa.core.errors import BuildError
from kalfa.std.common import positive_int


EMPTY = numpy.zeros(0, dtype="int64")


def positive(value, what):
    return positive_int(value, f"window: {what}")


def flat(values):
    values = torch.as_tensor(values)
    if values.ndim == 2 and values.shape[1] == 1:
        return values[:, 0]
    return values


def codes_of(groups, earlier, count, before):
    if groups is None:
        return numpy.zeros(count, dtype="int64"), numpy.zeros(before, dtype="int64")
    current = numpy.asarray(groups)
    if len(current) != count:
        raise BuildError(f"window: {len(current)} group values for {count} rows")
    if earlier is None or not before:
        codes, _ = pandas.factorize(current, sort=False, use_na_sentinel=False)
        return codes.astype("int64"), numpy.full(before, -1, dtype="int64")
    earlier = numpy.asarray(earlier)
    if len(earlier) != before:
        raise BuildError(f"window: {len(earlier)} group values for {before} context rows")
    codes, _ = pandas.factorize(numpy.concatenate([earlier, current]), sort=False, use_na_sentinel=False)
    codes = codes.astype("int64")
    return codes[before:], codes[:before]


def members_of(codes):
    if not len(codes):
        return {}
    order = numpy.argsort(codes, kind="stable")
    ordered = codes[order]
    cuts = numpy.flatnonzero(ordered[1:] != ordered[:-1]) + 1
    keys = ordered[numpy.concatenate([[0], cuts]).astype("int64")]
    return dict(zip(keys.tolist(), numpy.split(order.astype("int64"), cuts)))


def gather(values, earlier, layout, before):
    index = torch.from_numpy(layout)
    out = values.new_zeros((len(layout), *values.shape[1:]))
    late = index >= before
    out[late] = values[index[late] - before]
    if earlier is not None and tuple(earlier.shape[1:]) == tuple(values.shape[1:]):
        out[~late] = earlier[index[~late]].to(values.dtype)
    return out


class WindowDataset(Dataset):
    def __init__(self, x, targets, rows, starts, size, horizon):
        self.x = x
        self.targets = targets
        self.rows = rows
        self.starts = starts
        self.size = size
        self.horizon = horizon

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, position):
        start = int(self.starts[position])
        end = start + self.size
        item = {"x": self.x[start:end]}
        for name, values in self.targets.items():
            item[name] = values[end:end + self.horizon]
        return item


class Window:
    def __init__(self, size, horizon=1, stride=1, context=True, group=None):
        self.size = positive(size, "size")
        self.horizon = positive(horizon, "horizon")
        self.stride = positive(stride, "stride")
        self.context = bool(context)
        self.group = group

    def dataset(self, x, targets, rows, groups=None, context=None):
        x = torch.as_tensor(x, dtype=torch.float32)
        count = int(x.shape[0])
        rows = numpy.asarray(rows)
        if len(rows) != count:
            raise BuildError(f"window: {len(rows)} row ids for {count} rows")
        targets = {name: flat(values) for name, values in (targets or {}).items()}
        earlier_x, earlier_targets, earlier_groups = x[:0], {}, None
        if self.context and context is not None:
            earlier_x, earlier_targets, _, earlier_groups = context
            earlier_x = torch.as_tensor(earlier_x, dtype=torch.float32)
            earlier_targets = {name: flat(values) for name, values in (earlier_targets or {}).items()}
            if tuple(earlier_x.shape[1:]) != tuple(x.shape[1:]):
                raise BuildError(f"window: the context rows have the shape {tuple(earlier_x.shape[1:])}, the rows "
                                 f"{tuple(x.shape[1:])}")
        before = int(earlier_x.shape[0])
        current, previous = codes_of(groups, earlier_groups, count, before)
        tails = members_of(previous)
        pieces, starts, firsts = [], [], []
        length = 0
        for code, members in members_of(current).items():
            tail = tails.get(code, EMPTY)[-self.size:]
            offset = len(tail)
            total = offset + len(members)
            ends = numpy.arange(max(self.size, offset), total - self.horizon + 1, self.stride, dtype="int64")
            if not len(ends):
                continue
            pieces.append(numpy.concatenate([tail, members + before]))
            starts.append(ends - self.size + length)
            firsts.append(members[ends - offset])
            length += total
        layout = numpy.concatenate(pieces) if pieces else EMPTY
        starts = numpy.concatenate(starts) if starts else EMPTY
        firsts = numpy.concatenate(firsts) if firsts else EMPTY
        order = numpy.argsort(firsts, kind="stable")
        windows = gather(x, earlier_x, layout, before)
        horizons = {name: gather(values, earlier_targets.get(name), layout, before)
                    for name, values in targets.items()}
        return WindowDataset(windows, horizons, rows[firsts[order]], starts[order], self.size, self.horizon)


register("window", "kalfa.std.data.window.Window", slot="value",
         doc="turns the rows of a table into windows: x of the last size rows, each target over the next horizon "
             "rows; every window keeps to one value of the group column; with context the first windows of a set "
             "reach back into the previous set")
