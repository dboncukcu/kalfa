import math
from dataclasses import dataclass


@dataclass
class Choices:
    values: list


@dataclass
class Range:
    low: float
    high: float
    log: bool = False
    integer: bool = False
    steps: int | None = None


RANGE_KEYS = ("low", "high", "log", "int", "steps")


def parse_range(name, entry):
    unknown = [key for key in entry if key not in RANGE_KEYS]
    if unknown or "low" not in entry or "high" not in entry:
        raise ValueError(f"sweep.space.{name}: a range is {{low, high, log, int, steps}}, got {sorted(entry)}")
    low, high = entry["low"], entry["high"]
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in (low, high)) \
            or not low < high:
        raise ValueError(f"sweep.space.{name}: low must be a number below high")
    if entry.get("log") and low <= 0:
        raise ValueError(f"sweep.space.{name}: a log range needs low > 0")
    steps = entry.get("steps")
    return Range(float(low), float(high), bool(entry.get("log", False)), bool(entry.get("int", False)),
                 None if steps is None else int(steps))


def parse_space(mapping):
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("sweep.space maps param names to a list of choices or a range {low, high, log, int, steps}")
    space = {}
    for name, entry in mapping.items():
        if isinstance(entry, list):
            if not entry:
                raise ValueError(f"sweep.space.{name}: the list of choices is empty")
            space[name] = Choices(list(entry))
        elif isinstance(entry, dict):
            space[name] = parse_range(name, entry)
        else:
            raise ValueError(f"sweep.space.{name}: a list of choices or a range mapping")
    return space


def grid_values(name, entry):
    if isinstance(entry, Choices):
        return list(entry.values)
    if entry.steps is None or entry.steps < 2:
        raise ValueError(f"sweep.space.{name}: grid needs a list of choices or a range with steps >= 2")
    return [value_at(entry, position / (entry.steps - 1)) for position in range(entry.steps)]


def value_at(entry, fraction):
    if entry.log:
        value = math.exp(math.log(entry.low) + fraction * (math.log(entry.high) - math.log(entry.low)))
    else:
        value = entry.low + fraction * (entry.high - entry.low)
    return int(round(value)) if entry.integer else float(value)


def sample(entry, fraction):
    if isinstance(entry, Choices):
        return entry.values[min(int(fraction * len(entry.values)), len(entry.values) - 1)]
    return value_at(entry, fraction)


def point_from_fractions(space, fractions):
    return {name: sample(entry, float(fraction)) for (name, entry), fraction in zip(space.items(), fractions)}


class Strategy:
    deterministic = False

    def total(self, space: dict) -> int:
        raise NotImplementedError

    def point(self, space: dict, index: int) -> dict:
        raise NotImplementedError
