import numpy
import torch

from kalfa.registration import lego
from kalfa.std.feed.base import Dataset, IterableDataset


class TableDataset(Dataset):
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


class StreamDataset(IterableDataset):
    def __init__(self, frame):
        self.frame = frame
        self.inputs = ["x"]
        self.targets = list(frame.targets)
        self.last_rows = []

    def items_of(self, index, data):
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
            for row, item in self.items_of(index, data):
                count += 1
                if not self.shuffle:
                    rows.append(row)
                    yield item
                    continue
                buffer.append((row, item))
                if len(buffer) >= self.buffer:
                    row, item = pop_random(buffer)
                    rows.append(row)
                    yield item
        while buffer:
            row, item = pop_random(buffer)
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


def pop_random(buffer):
    pick = int(torch.randint(len(buffer), (1,)))
    buffer[pick], buffer[-1] = buffer[-1], buffer[pick]
    return buffer.pop()


def tensor_of(value, dtype):
    if isinstance(value, torch.Tensor):
        return value
    if dtype == "int64":
        return torch.as_tensor(int(value), dtype=torch.int64)
    if dtype == "bool":
        return torch.as_tensor(bool(value))
    return torch.as_tensor(numpy.asarray(value, dtype="float32"))


class SampleDataset(Dataset):
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
            for preprocessor in self.frame.chains.get(name, []):
                value = preprocessor.apply(value)
            out[name] = tensor_of(value, self.frame.dataset.dtypes.get(name))
        return out

    def rows(self):
        return numpy.asarray(self.frame.index)

    def labels(self, name):
        values = numpy.asarray(self.frame.dataset.column(name))
        for preprocessor in self.frame.chains.get(name, []):
            values = numpy.asarray(preprocessor.apply(values))
        return torch.as_tensor(values)


@lego("/feed/kalfa/table", alias="table",
      description="Feature columns as one tensor x and target fields by name; Dataset fields by name")
def table(frame, frames=None):
    if frame.stream is not None:
        return StreamDataset(frame)
    if frame.dataset is not None:
        return SampleDataset(frame)
    return TableDataset(frame)
