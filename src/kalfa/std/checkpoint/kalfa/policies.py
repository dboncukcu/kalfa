import math

from kalfa.std.checkpoint.base import Policy


class Best(Policy):
    def __init__(self, monitor, mode="min", last=True):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be min or max, got {mode!r}")
        self.monitor = monitor
        self.mode = mode
        self.last = bool(last)
        self.best = None

    def tags(self, metrics):
        trailing = ["last"] if self.last else []
        value = (metrics or {}).get(self.monitor)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return trailing
        value = float(value)
        improved = self.best is None or (value > self.best if self.mode == "max" else value < self.best)
        if improved:
            self.best = value
            return ["best", *trailing]
        return trailing

    def state(self):
        return {"best": self.best}

    def restore(self, state):
        if state and "best" in state:
            self.best = state["best"]


class Last(Policy):
    def tags(self, metrics):
        return ["last"]


class Snapshot(Policy):
    def __init__(self, every):
        self.every = int(every)
        self.seen = 0

    def tags(self, metrics):
        self.seen += 1
        tags = ["last"]
        if self.seen % self.every == 0:
            tags.append(f"snapshot_{self.seen}")
        return tags

    def state(self):
        return {"seen": self.seen}

    def restore(self, state):
        if state and "seen" in state:
            self.seen = int(state["seen"])
