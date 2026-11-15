import math
from dataclasses import dataclass

import numpy

from kalfa.core.errors import BuildError


SHAPE = ("weights maps a batch field name to a weight column or to a mapping of pandas conditions to weights, such as "
         "{w: {\"price < 5\": 5}}")


@dataclass
class Weight:
    name: str
    column: str | None = None
    rules: tuple = ()

    def source(self):
        if self.column is not None:
            return self.column
        return {text: value for text, value in self.rules}

    def detail(self):
        if self.column is not None:
            return f"column {self.column}"
        return "; ".join([f"{text}: {value:g}" for text, value in self.rules] + ["else 1"])

    def values(self, frame):
        where = f"weights.{self.name}"
        if self.column is not None:
            found = column_values(frame, self.column, where)
        else:
            found = numpy.ones(len(frame), dtype="float64")
            free = numpy.ones(len(frame), dtype=bool)
            for text, value in self.rules:
                hit = mask_of(frame, text, where) & free
                found[hit] = value
                free &= ~hit
        bad = int((~numpy.isfinite(found)).sum())
        if bad:
            raise BuildError(f"{where}: {bad} rows have a weight that is not a finite number")
        return found


def column_values(frame, column, where):
    if column not in frame.columns:
        raise BuildError(f"{where}: the table has no column {column!r}; the columns are {list(frame.columns)}")
    try:
        return numpy.asarray(frame[column].to_numpy(), dtype="float64")
    except (TypeError, ValueError) as error:
        raise BuildError(f"{where}: the column {column!r} is not numeric") from error


def mask_of(frame, text, where):
    try:
        found = frame.eval(text)
    except Exception as error:
        raise BuildError(f"{where}: the condition {text!r} failed: {error}") from error
    mask = numpy.asarray(found)
    if mask.dtype != bool or mask.shape != (len(frame),):
        raise BuildError(f"{where}: the condition {text!r} must give true or false for every row, like 'price < 5'")
    return mask


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def rules_of(name, item):
    rules = []
    for text, value in item.items():
        if not isinstance(text, str) or not text.strip():
            raise BuildError(f"weights.{name}: a condition is a pandas expression in quotes such as \"price < 5\", "
                             f"got {text!r}")
        if not number(value):
            raise BuildError(f"weights.{name}: the weight of {text!r} must be a finite number, got {value!r}")
        rules.append((text, float(value)))
    return tuple(rules)


def weights_of(spec, targets=None, drop=None):
    if spec is None:
        return []
    if not isinstance(spec, dict):
        raise BuildError(f"{SHAPE}; got {type(spec).__name__}")
    taken = {str(name) for name in targets or ()}
    dropped = {str(name) for name in drop or ()} if isinstance(drop, (list, tuple, set)) else set()
    found = []
    for name, item in spec.items():
        if not isinstance(name, str) or not name:
            raise BuildError(f"weights: a batch field name is text, got {name!r}")
        if name == "x":
            raise BuildError("weights: a weight cannot be named x; x is the feature tensor")
        if name in taken:
            raise BuildError(f"weights: {name!r} is a target; give the weight field its own name")
        if name in dropped:
            raise BuildError(f"weights: {name!r} is in drop; a weight field cannot be dropped")
        if isinstance(item, str) and item:
            found.append(Weight(name, column=item))
        elif isinstance(item, dict) and item:
            found.append(Weight(name, rules=rules_of(name, item)))
        else:
            raise BuildError(f"weights.{name}: {SHAPE}; got {item!r}")
    return found


def scale_of(weight, values):
    mean = float(numpy.mean(values)) if len(values) else math.nan
    if not mean > 0:
        raise BuildError(f"weights.{weight.name}: the train weights have the mean {mean:g}; it must be positive to "
                         "scale them to mean one")
    return 1.0 / mean
