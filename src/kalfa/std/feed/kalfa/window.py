import numpy
import torch

from kalfa.registration import lego
from kalfa.std.feed.base import Dataset


SET_ORDER = ("train", "valid", "test")


def previous_frames(frame, frames):
    if not frames or frame.set not in SET_ORDER:
        return []
    position = SET_ORDER.index(frame.set)
    return [frames[name] for name in SET_ORDER[:position] if frames.get(name) is not None and len(frames[name])]


class WindowDataset(Dataset):
    def __init__(self, frame, size, horizon, context=None, group=None):
        self.frame = frame
        self.window = int(size)
        self.horizon = int(horizon)
        self.inputs = ["x"]
        self.targets = list(frame.targets)
        windows = []
        labels = {name: [] for name in self.targets}
        row_ids = []
        for key, part, tail in self.series_of(frame, context, group):
            x = numpy.concatenate([tail["x"], part["x"]]) if tail is not None else part["x"]
            offset = len(tail["x"]) if tail is not None else 0
            for end in range(max(self.window, offset), len(x) - self.horizon + 1):
                windows.append(x[end - self.window:end])
                for name in self.targets:
                    values = part["targets"][name]
                    labels[name].append(values[end - offset:end - offset + self.horizon])
                row_ids.append(part["rows"][end - offset])
        self.x = torch.from_numpy(numpy.stack(windows).astype("float32")) if windows else \
            torch.zeros((0, self.window, len(frame.features)), dtype=torch.float32)
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
    def parts_of(frame, group):
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

    def series_of(self, frame, context, group):
        tails = {}
        for previous in context or []:
            for key, part in self.parts_of(previous, group):
                earlier = tails.get(key)
                joined = numpy.concatenate([earlier["x"], part["x"]]) if earlier is not None else part["x"]
                tails[key] = {"x": joined[-self.window:]}
        for key, part in self.parts_of(frame, group):
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
