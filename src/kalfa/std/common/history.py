import math
from pathlib import Path

from kalfa.std.common.files import append_line, read_lines


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_series(key, value):
    return "/" in key and not key.startswith("lr/") and is_number(value)


class History:
    def __init__(self, lines=()):
        self.lines = list(lines or [])

    @classmethod
    def read(cls, record):
        return cls(read_lines(Path(record) / "history.jsonl"))

    @staticmethod
    def line(metrics, counters, optimizers, rules, seconds=None, minimizes=None):
        found = {"turn": int((counters or {}).get("turn", 0)),
                 "global_step": int((counters or {}).get("global_step", 0))}
        found.update(metrics or {})
        for name, optimizer in (optimizers or {}).items():
            found[f"lr/{name}"] = optimizer.lr()
        for name, loss in (minimizes or {}).items():
            found[f"minimizes/{name}"] = loss
        if seconds is not None:
            found["seconds"] = round(float(seconds), 3)
        found["rules"] = list((rules or {}).get("fired") or [])
        return found

    @classmethod
    def read_steps(cls, record):
        return cls(read_lines(Path(record) / "steps.jsonl"))

    @staticmethod
    def append(record, line):
        History.append_steps(record, [line], "history.jsonl")

    @staticmethod
    def append_steps(record, lines, name="steps.jsonl"):
        for line in lines:
            append_line(Path(record) / name, line)

    def __len__(self):
        return len(self.lines)

    def __iter__(self):
        return iter(self.lines)

    def __getitem__(self, position):
        return self.lines[position]

    def __eq__(self, other):
        return isinstance(other, History) and self.lines == other.lines

    def positions(self, key="turn"):
        return [line.get(key, position + 1) for position, line in enumerate(self.lines)]

    def rates(self):
        found = {}
        for line in self.lines:
            for key, value in line.items():
                if key.startswith("lr/") and is_number(value):
                    found.setdefault(key, []).append(value)
        return found

    def series(self, names=None):
        found = {}
        for line in self.lines:
            for key, value in line.items():
                if is_series(key, value):
                    found.setdefault(key, []).append(value)
        if names:
            rates = self.rates()
            missing = [name for name in names if name not in found and name not in rates]
            if missing:
                raise ValueError(f"{missing} are not in the history; the series are {sorted(found)} and the "
                                 f"rates {sorted(rates)}")
            found = {name: found[name] if name in found else rates[name] for name in names}
        return found

    def last(self, prefixes=("test/",)):
        if not self.lines:
            return {}
        return {key: value for key, value in self.lines[-1].items() if is_number(value) and key.startswith(prefixes)}

    def best(self, monitor, mode="min", at="best"):
        lines = [line for line in self.lines if is_number(line.get(monitor)) and not math.isnan(line[monitor])]
        if not lines:
            raise ValueError(f"the history has no value for {monitor!r}")
        line = lines[-1] if at == "last" else (min if mode == "min" else max)(lines, key=lambda entry: entry[monitor])
        return float(line[monitor]), int(line.get("turn", len(lines)))
