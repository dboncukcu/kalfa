import copy
import math

import numpy

from kalfa.core.errors import BuildError
from kalfa.records import reader


RANGES = ("log", "float", "int")
SPACE_HINT = "a param maps to a list of values or to {log: [a, b]}, {float: [a, b]} or {int: [a, b]}"


def is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def spread(kind, bounds, steps):
    low, high = bounds
    grid = numpy.geomspace(low, high, steps) if kind == "log" else numpy.linspace(low, high, steps)
    if kind == "int":
        return list(dict.fromkeys(round(float(value)) for value in grid))
    return [float(f"{float(value):.12g}") for value in grid]


def draw(generator, kind, values):
    if kind == "list":
        return copy.deepcopy(values[int(generator.integers(len(values)))])
    low, high = values
    if kind == "int":
        return int(generator.integers(low, high + 1))
    if kind == "log":
        return float(min(max(math.exp(generator.uniform(math.log(low), math.log(high))), low), high))
    return float(generator.uniform(low, high))


def space_values(space):
    if not isinstance(space, dict) or not space:
        raise BuildError(f"sweep.space must be a mapping of params to values; {SPACE_HINT}")
    return {str(name): dimension(name, entry) for name, entry in space.items()}


def dimension(name, entry):
    if isinstance(entry, (list, tuple)):
        if not entry:
            raise BuildError(f"sweep.space.{name}: the list of values is empty")
        return "list", list(entry)
    if isinstance(entry, dict) and len(entry) == 1:
        kind, bounds = next(iter(entry.items()))
        if kind in RANGES:
            return kind, bounds_of(name, kind, bounds)
    raise BuildError(f"sweep.space.{name}: {entry!r} is neither a list of values nor a range; {SPACE_HINT}")


def bounds_of(name, kind, bounds):
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2 or not all(map(reader.is_finite, bounds)):
        raise BuildError(f"sweep.space.{name}: {kind} takes two numbers [low, high], got {bounds!r}")
    low, high = bounds
    if kind == "int" and not (is_int(low) and is_int(high)):
        raise BuildError(f"sweep.space.{name}: int takes two whole numbers [low, high], got {list(bounds)!r}")
    if low > high:
        raise BuildError(f"sweep.space.{name}: {kind} [{low}, {high}] has its low end above its high end")
    if kind == "log" and low <= 0:
        raise BuildError(f"sweep.space.{name}: log [{low}, {high}] needs a low end above zero")
    if kind == "int":
        return [int(low), int(high)]
    return [float(low), float(high)]
