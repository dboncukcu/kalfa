import math


RANGE_KEYS = ("low", "high", "log", "int", "steps")


class Choices:
    def __init__(self, values):
        self.values = list(values)


class Range:
    def __init__(self, low, high, log=False, integer=False, steps=None):
        self.low = float(low)
        self.high = float(high)
        self.log = bool(log)
        self.integer = bool(integer)
        self.steps = None if steps is None else int(steps)


def parse_space(mapping):
    """The space of a sweep section: name -> Choices or Range, validated."""
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("sweep.space maps param names to a list of choices or a range {low, high, log, int, steps}")
    space = {}
    for name, entry in mapping.items():
        if isinstance(entry, list):
            if not entry:
                raise ValueError(f"sweep.space.{name}: the list of choices is empty")
            space[name] = Choices(entry)
        elif isinstance(entry, dict):
            unknown = [key for key in entry if key not in RANGE_KEYS]
            if unknown or "low" not in entry or "high" not in entry:
                raise ValueError(f"sweep.space.{name}: a range is {{low, high, log, int, steps}}, got {sorted(entry)}")
            low, high = entry["low"], entry["high"]
            if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in (low, high)) \
                    or not low < high:
                raise ValueError(f"sweep.space.{name}: low must be a number below high")
            if entry.get("log") and low <= 0:
                raise ValueError(f"sweep.space.{name}: a log range needs low > 0")
            space[name] = Range(low, high, entry.get("log", False), entry.get("int", False), entry.get("steps"))
        else:
            raise ValueError(f"sweep.space.{name}: a list of choices or a range mapping")
    return space


def grid_values(name, entry):
    """The values a grid enumerates for one entry: the choices, or steps points of a range."""
    if isinstance(entry, Choices):
        return list(entry.values)
    if entry.steps is None or entry.steps < 2:
        raise ValueError(f"sweep.space.{name}: grid needs a list of choices or a range with steps >= 2")
    values = []
    for position in range(entry.steps):
        fraction = position / (entry.steps - 1)
        values.append(value_at(entry, fraction))
    return values


def value_at(entry, fraction):
    if entry.log:
        value = math.exp(math.log(entry.low) + fraction * (math.log(entry.high) - math.log(entry.low)))
    else:
        value = entry.low + fraction * (entry.high - entry.low)
    return int(round(value)) if entry.integer else float(value)


def sample(entry, fraction):
    """The value at a fraction of [0, 1): a choice by position, or a point of the range."""
    if isinstance(entry, Choices):
        return entry.values[min(int(fraction * len(entry.values)), len(entry.values) - 1)]
    return value_at(entry, fraction)


def point_from_fractions(space, fractions):
    return {name: sample(entry, float(fraction)) for (name, entry), fraction in zip(space.items(), fractions)}


class Strategy:
    deterministic = False

    def total(self, space):
        raise NotImplementedError

    def point(self, space, index):
        raise NotImplementedError
