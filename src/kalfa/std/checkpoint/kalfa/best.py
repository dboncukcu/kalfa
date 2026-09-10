import math

from kalfa.registration import lego
from kalfa.std.checkpoint.base import Policy


class Best(Policy):
    def __init__(self, monitor, mode):
        self.monitor = monitor
        self.mode = mode
        self.best = None

    def tags(self, metrics):
        value = (metrics or {}).get(self.monitor)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return ["last"]
        value = float(value)
        improved = self.best is None or (value > self.best if self.mode == "max" else value < self.best)
        if improved:
            self.best = value
            return ["best", "last"]
        return ["last"]

    def state(self):
        return {"best": self.best}

    def restore(self, state):
        if state and "best" in state:
            self.best = state["best"]


@lego("/checkpoint/kalfa/best", alias="best",
      description="Write best.pt when the monitored value improves and last.pt every turn")
def best(monitor, mode="min"):
    if mode not in ("min", "max"):
        raise ValueError(f"mode must be min or max, got {mode!r}")
    return Best(monitor, mode)
