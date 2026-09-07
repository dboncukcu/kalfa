"""Feeds: legos that turn a preprocessed frame into a Dataset of batch fields."""

import numpy
import torch
from torch.utils.data import Dataset, IterableDataset

from ..registration import lego


SET_ORDER = ("train", "valid", "test")


class TableDataset(Dataset):
    """Rows of a table: the feature columns as one tensor ``x``, every target field under its own name."""

    def __init__(self, frame):
        self.frame = frame
        self.inputs = ["x"]
        self.targets = list(frame.targets)
        data = frame.data
        if len(frame.features):
            self.x = torch.from_numpy(numpy.array(data[frame.features].to_numpy(dtype="float32"), copy=True))
        else:
            self.x = torch.zeros((len(data), 0), dtype=torch.float32)
        self.fields = {}
        for name, columns in frame.targets.items():
            values = data[columns].to_numpy()
            if len(columns) == 1:
                values = values.reshape(-1)
            self.fields[name] = torch.from_numpy(numpy.array(values, copy=True))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        item = {"x": self.x[index]}
        for name, values in self.fields.items():
            item[name] = values[index]
        return item

    def rows(self):
        return numpy.asarray(self.frame.index)

    def labels(self, name):
        return self.fields[name]


def dataset_size(dataset):
    """The number of items of a dataset, counted by a pass when it has no length."""
    try:
        return len(dataset)
    except TypeError:
        return sum(1 for _ in dataset)


def sized(dataset):
    """The length of a dataset, None for a stream that has to be read to be counted."""
    try:
        return len(dataset)
    except TypeError:
        return None


class StreamDataset(IterableDataset):
    """Rows of a stream frame, one pass per iteration: the feature tensor ``x`` and every target by name; the train
    pass shuffles through a buffer; the row ids of the last full pass are kept for the prediction table."""

    def __init__(self, frame):
        self.frame = frame
        self.inputs = ["x"]
        self.targets = list(frame.targets)
        self.shuffle = False
        self.buffer = 4096
        self.last_rows = []

    def _items(self, index, data):
        if len(self.frame.features):
            x = torch.from_numpy(numpy.array(data[self.frame.features].to_numpy(dtype="float32"), copy=True))
        else:
            x = torch.zeros((len(data), 0), dtype=torch.float32)
        fields = {}
        for name, columns in self.frame.targets.items():
            values = data[columns].to_numpy()
            if len(columns) == 1:
                values = values.reshape(-1)
            fields[name] = torch.from_numpy(numpy.array(values, copy=True))
        for position in range(len(data)):
            item = {"x": x[position]}
            for name, values in fields.items():
                item[name] = values[position]
            yield int(index[position]), item

    def __iter__(self):
        rows = []
        buffer = []
        count = 0
        for index, data in self.frame.stream.chunks():
            for row, item in self._items(index, data):
                count += 1
                if not self.shuffle:
                    rows.append(row)
                    yield item
                    continue
                buffer.append((row, item))
                if len(buffer) >= self.buffer:
                    row, item = _pop_random(buffer)
                    rows.append(row)
                    yield item
        while buffer:
            row, item = _pop_random(buffer)
            rows.append(row)
            yield item
        if count == 0 and self.frame.set == "train":
            raise ValueError("the train stream yields no rows: the source, the window or the filters leave nothing "
                             "(the lazy set reports an empty train set at its first pass)")
        self.last_rows = rows

    def rows(self):
        return numpy.asarray(self.last_rows, dtype="int64")

    def labels(self, name):
        raise ValueError("a lazy set cannot be counted: balanced and class_weights need a table source")


def _pop_random(buffer):
    pick = int(torch.randint(len(buffer), (1,)))
    buffer[pick], buffer[-1] = buffer[-1], buffer[pick]
    return buffer.pop()


def _as_tensor(value, dtype):
    if isinstance(value, torch.Tensor):
        return value
    if dtype == "int64":
        return torch.as_tensor(int(value), dtype=torch.int64)
    if dtype == "bool":
        return torch.as_tensor(bool(value))
    return torch.as_tensor(numpy.asarray(value, dtype="float32"))


class SampleDataset(Dataset):
    """Items of a Dataset source: every field under its own name, the chains applied per item."""

    def __init__(self, frame):
        self.frame = frame
        self.targets = [name for name in frame.fields if name in frame.targets]
        self.inputs = [name for name in frame.fields if name not in frame.targets]

    def __len__(self):
        return len(self.frame.dataset)

    def __getitem__(self, index):
        item = self.frame.dataset[index]
        out = {}
        for name in self.frame.fields:
            value = item[name]
            for obj in self.frame.chains.get(name, []):
                value = obj.apply(value)
            out[name] = _as_tensor(value, self.frame.dataset.dtypes.get(name))
        return out

    def rows(self):
        return numpy.asarray(self.frame.index)

    def labels(self, name):
        values = numpy.asarray(self.frame.dataset.column(name))
        for obj in self.frame.chains.get(name, []):
            values = numpy.asarray(obj.apply(values))
        return torch.as_tensor(values)


@lego("/feed/kalfa/table", alias="table",
            description="Feature columns as one tensor x and target fields by name; Dataset fields by name")
def table(frame, frames=None):
    if frame.stream is not None:
        return StreamDataset(frame)
    if frame.dataset is not None:
        return SampleDataset(frame)
    return TableDataset(frame)


def previous_frames(frame, frames):
    """The sets before this one in train, valid, test order; their tail is the window context at the boundary."""
    if not frames or frame.set not in SET_ORDER:
        return []
    position = SET_ORDER.index(frame.set)
    return [frames[name] for name in SET_ORDER[:position] if frames.get(name) is not None and len(frames[name])]


class WindowDataset(Dataset):
    """Sliding windows of ``size`` steps of the features and the next ``horizon`` steps of every target field.

    With a group column every group is its own series and no window crosses groups; with ``context`` (the earlier
    sets in train, valid, test order) the last ``size`` rows of the same group before this set precede the first
    windows, so the set boundary loses no targets. ``rows`` gives the source row id of the first target step of
    every window.
    """

    def __init__(self, frame, size, horizon, context=None, group=None):
        self.frame = frame
        self.size = int(size)
        self.horizon = int(horizon)
        self.inputs = ["x"]
        self.targets = list(frame.targets)
        windows = []
        labels = {name: [] for name in self.targets}
        row_ids = []
        for key, part, tail in self._series(frame, context, group):
            x = numpy.concatenate([tail["x"], part["x"]]) if tail is not None else part["x"]
            offset = len(tail["x"]) if tail is not None else 0
            for end in range(max(self.size, offset), len(x) - self.horizon + 1):
                windows.append(x[end - self.size:end])
                for name in self.targets:
                    values = part["targets"][name]
                    labels[name].append(values[end - offset:end - offset + self.horizon])
                row_ids.append(part["rows"][end - offset])
        self.x = torch.from_numpy(numpy.stack(windows).astype("float32")) if windows else \
            torch.zeros((0, self.size, len(frame.features)), dtype=torch.float32)
        self.fields = {}
        for name in self.targets:
            if labels[name]:
                stacked = numpy.stack(labels[name])
                if stacked.ndim == 3 and stacked.shape[2] == 1:
                    stacked = stacked[:, :, 0]
                self.fields[name] = torch.from_numpy(numpy.ascontiguousarray(stacked))
            else:
                self.fields[name] = torch.zeros((0, self.horizon), dtype=torch.float32)
        self.row_ids = numpy.asarray(row_ids)

    @staticmethod
    def _parts(frame, group):
        data = frame.data
        if group is None or frame.extra is None or group not in frame.extra.columns:
            keys = [None]
            masks = [numpy.ones(len(data), dtype=bool)]
        else:
            column = frame.extra[group].to_numpy()
            keys = list(dict.fromkeys(column.tolist()))
            masks = [column == key for key in keys]
        for key, mask in zip(keys, masks):
            part = data[mask]
            yield key, {"x": part[frame.features].to_numpy(dtype="float32"),
                        "targets": {name: part[columns].to_numpy() for name, columns in frame.targets.items()},
                        "rows": numpy.asarray(part.index)}

    def _series(self, frame, context, group):
        tails = {}
        for previous in context or []:
            for key, part in self._parts(previous, group):
                earlier = tails.get(key)
                joined = numpy.concatenate([earlier["x"], part["x"]]) if earlier is not None else part["x"]
                tails[key] = {"x": joined[-self.size:]}
        for key, part in self._parts(frame, group):
            yield key, part, tails.get(key)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        item = {"x": self.x[index]}
        for name, values in self.fields.items():
            item[name] = values[index]
        return item

    def rows(self):
        return self.row_ids


@lego("/feed/kalfa/window", alias="window", refs={"group": "column"},
            description="Windows of size steps and the next horizon steps of the targets; context takes the tail "
                        "of the previous set at the split boundary, group keeps series apart")
def window(frame, frames, size, horizon, context=False, group=None):
    if frame.stream is not None:
        raise ValueError("window needs a table in memory; the lazy set has the table feed only")
    return WindowDataset(frame, size, horizon, previous_frames(frame, frames) if context else None, group)


class TokenDataset(Dataset):
    """Non overlapping windows of seq_len tokens over the token stream of the set's lines, targets shifted by one."""

    def __init__(self, frame, seq_len):
        self.frame = frame
        self.seq_len = int(seq_len)
        self.inputs = ["input_ids"]
        self.targets = ["targets"]
        chains = frame.chains.get("text", [])
        newline = None
        for obj in chains:
            if hasattr(obj, "encode"):
                newline = int(obj.encode("\n")[0])
        pieces = []
        for position in range(len(frame.dataset)):
            value = frame.dataset[position]["text"]
            for obj in chains:
                value = obj.apply(value)
            pieces.append(numpy.asarray(value, dtype="int64").reshape(-1))
            if newline is not None:
                pieces.append(numpy.array([newline], dtype="int64"))
        stream = numpy.concatenate(pieces) if pieces else numpy.zeros(0, dtype="int64")
        count = max((len(stream) - 1) // self.seq_len, 0)
        starts = numpy.arange(count) * self.seq_len
        self.input_ids = torch.from_numpy(numpy.stack([stream[start:start + self.seq_len] for start in starts])) \
            if count else torch.zeros((0, self.seq_len), dtype=torch.int64)
        self.target_ids = torch.from_numpy(numpy.stack([stream[start + 1:start + self.seq_len + 1] for start in starts])) \
            if count else torch.zeros((0, self.seq_len), dtype=torch.int64)
        self.starts = starts

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return {"input_ids": self.input_ids[index], "targets": self.target_ids[index]}

    def rows(self):
        return self.starts

    def labels(self, name):
        return self.target_ids


@lego("/feed/kalfa/next_token", alias="next_token",
            description="input_ids and targets windows of seq_len tokens over the set's token stream")
def next_token(frame, frames, seq_len):
    return TokenDataset(frame, seq_len)
