import math

from kalfa.core.aliases import register


def fraction(update, steps):
    return min(max(float(update) / float(steps), 0.0), 1.0)


class Warmup:
    def __init__(self, steps):
        self.steps = int(steps)

    def __call__(self, update):
        if self.steps <= 0:
            return 1.0
        return fraction(update, self.steps)


class Cosine:
    def __init__(self, steps, warmup=0, floor=0.0):
        self.steps = int(steps)
        self.warmup = int(warmup)
        self.floor = float(floor)

    def __call__(self, update):
        update = float(update)
        if self.warmup > 0 and update < self.warmup:
            return fraction(update, self.warmup)
        if self.steps <= self.warmup:
            return self.floor
        progress = fraction(update - self.warmup, self.steps - self.warmup)
        return self.floor + (1.0 - self.floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


class LinearWarmup:
    def __init__(self, start, end, steps):
        self.start = float(start)
        self.end = float(end)
        self.steps = int(steps)

    def __call__(self, update):
        if self.steps <= 0:
            return self.end
        return self.start + (self.end - self.start) * fraction(update, self.steps)


class StepDecay:
    def __init__(self, every, gamma=0.1):
        self.every = int(every)
        self.gamma = float(gamma)

    def __call__(self, update):
        if self.every <= 0:
            return 1.0
        return self.gamma ** (int(update) // self.every)


SCHEDULES = (
    ("warmup", "kalfa.std.schedules.Warmup", "a factor rising linearly from 0 to 1 over steps updates, then 1"),
    ("cosine", "kalfa.std.schedules.Cosine", "linear warmup over warmup updates, then cosine decay from 1 to "
                                             "floor at steps updates"),
    ("linear_warmup", "kalfa.std.schedules.LinearWarmup", "a value moving linearly from start to end over steps "
                                                          "updates, then end; a loss weight or an lr factor"),
    ("step_decay", "kalfa.std.schedules.StepDecay", "a factor multiplied by gamma every every updates"),
)


def register_names():
    for name, target, doc in SCHEDULES:
        register(name, target, slot="schedule", doc=doc)


register_names()
