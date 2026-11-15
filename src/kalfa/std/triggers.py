import math
import time

from kalfa.core.aliases import register
from kalfa.core.protocols import Trigger


def latest(ctx, key):
    value = (getattr(ctx, "latest", None) or {}).get(key)
    if value is None:
        return None
    number = float(value)
    return None if math.isnan(number) else number


class AfterEpoch(Trigger):
    def __init__(self, at):
        self.at = int(at)

    def fires(self, ctx):
        return ctx.turn >= self.at


class Every(Trigger):
    def __init__(self, epochs):
        self.epochs = int(epochs)

    def fires(self, ctx):
        return self.epochs > 0 and ctx.turn % self.epochs == 0


class Plateau(Trigger):
    persist = ("best", "wait")

    def __init__(self, watch, patience, mode="min", min_delta=0.0):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be min or max, got {mode!r}")
        self.watch = str(watch)
        self.patience = int(patience)
        self.mode = mode
        self.min_delta = float(min_delta)
        self.best = None
        self.wait = 0

    def improved(self, value):
        if self.best is None:
            return True
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def reset(self):
        self.best = None
        self.wait = 0

    def fires(self, ctx):
        value = latest(ctx, self.watch)
        if value is None:
            return False
        if self.improved(value):
            self.best = value
            self.wait = 0
        else:
            self.wait += 1
        return self.wait >= self.patience

    def describe(self):
        return f"{self.watch} plateau {self.patience}"


class Above(Trigger):
    def __init__(self, watch, value):
        self.watch = str(watch)
        self.value = float(value)

    def fires(self, ctx):
        current = latest(ctx, self.watch)
        return current is not None and current > self.value

    def describe(self):
        return f"{self.watch} > {self.value}"


class Below(Trigger):
    def __init__(self, watch, value):
        self.watch = str(watch)
        self.value = float(value)

    def fires(self, ctx):
        current = latest(ctx, self.watch)
        return current is not None and current < self.value

    def describe(self):
        return f"{self.watch} < {self.value}"


class TimeBudget(Trigger):
    persist = ("started",)

    def __init__(self, minutes):
        self.minutes = float(minutes)
        self.started = None

    def reset(self):
        self.started = None

    def fires(self, ctx):
        now = time.time()
        if self.started is None:
            self.started = now
        return (now - self.started) >= self.minutes * 60.0

    def describe(self):
        return f"after {self.minutes} minutes"


TRIGGERS = (
    ("after_epoch", "kalfa.std.triggers.AfterEpoch", "fires from the turn whose count reaches at (absolute, so a "
                                                     "resumed run counts on)"),
    ("every", "kalfa.std.triggers.Every", "fires every epochs turns"),
    ("plateau", "kalfa.std.triggers.Plateau", "fires after patience turns without the watched series improving "
                                              "by min_delta; turns without a value are not counted"),
    ("above", "kalfa.std.triggers.Above", "fires when the watched series is above value"),
    ("below", "kalfa.std.triggers.Below", "fires when the watched series is below value"),
    ("time_budget", "kalfa.std.triggers.TimeBudget", "fires once minutes have passed since the first turn that "
                                                     "asked it"),
)


def register_names():
    for name, target, doc in TRIGGERS:
        register(name, target, slot="trigger", doc=doc)


register_names()
