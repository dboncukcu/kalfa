import fnmatch
from dataclasses import dataclass, field

import numpy
import torch

from kalfa.core import aliases
from kalfa.core.coerce import coerced
from kalfa.core.errors import BuildError, error
from kalfa.std.data import pre


def is_pattern(text):
    return any(character in text for character in "*?[")


def matches(column, pattern):
    return fnmatch.fnmatchcase(column, pattern) if is_pattern(pattern) else column == pattern


def matches_any(column, patterns):
    return any(matches(column, pattern) for pattern in patterns)


def specificity(pattern):
    return sum(1 for character in pattern if character not in "*?["), pattern.count("?")


def owners_of(columns, patterns):
    owners = {}
    for column in columns:
        claims = [pattern for pattern in patterns if matches(column, pattern)]
        if not claims:
            continue
        literal = [pattern for pattern in claims if not is_pattern(pattern)]
        if literal:
            owners[column] = literal[0]
            continue
        ranked = sorted(claims, key=specificity, reverse=True)
        if len(ranked) > 1 and specificity(ranked[0]) == specificity(ranked[1]):
            raise BuildError(f"column {column!r} matches the features patterns {ranked[0]!r} and {ranked[1]!r} "
                             f"with equal specificity; make one of them more specific")
        owners[column] = ranked[0]
    return owners


def hint_of(name):
    return aliases.suggestion(name) if isinstance(name, str) and not aliases.is_import_path(name) else None


def target_of(name, where=None):
    try:
        return aliases.import_target(aliases.resolve(name))
    except BuildError as failure:
        hint = hint_of(name)
        place = f"{where}: " if where else ""
        raise BuildError(f"{place}{failure}" + (f" ({hint})" if hint else "")) from None


def name_problems(spec, where, problems):
    for item in spec if isinstance(spec, (list, tuple)) else [spec]:
        if not isinstance(item, str):
            continue
        try:
            found = aliases.resolve(item)
        except BuildError as failure:
            problems.append(error("unknown_use", str(failure), where, hint_of(item)))
            continue
        try:
            aliases.import_target(found)
        except BuildError as failure:
            problems.append(error("import_failed", str(failure), where))


def step_of(spec, where):
    if isinstance(spec, str):
        return target_of(spec, where)()
    if isinstance(spec, dict):
        if "use" not in spec:
            raise BuildError(f"{where}: a preprocessor call needs use, got the keys {sorted(spec)}")
        params = {key: value for key, value in spec.items() if key != "use"}
        target = target_of(spec["use"], where)
        return target(**coerced(target, params))
    if hasattr(spec, "transform"):
        return spec
    raise BuildError(f"{where}: a preprocessor is a short name, an import path, a call or an object with "
                     f"transform, got {type(spec).__name__}")


def chain_of(spec, where):
    if spec is None:
        return []
    items = list(spec) if isinstance(spec, (list, tuple)) else [spec]
    return [step_of(item, where) for item in items]


def as_matrix(values):
    array = numpy.asarray(values)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim > 2:
        return array.reshape(len(array), -1)
    return array


def apply_step(step, values, where, fit):
    try:
        if fit and hasattr(step, "fit"):
            step.fit(values)
        return as_matrix(step.transform(values))
    except (ValueError, TypeError) as error:
        raise BuildError(f"{where}: {type(step).__name__} failed: {error}") from error


def names_after(step, names, width):
    if hasattr(step, "get_feature_names_out"):
        try:
            return [str(name) for name in step.get_feature_names_out(list(names))]
        except (ValueError, TypeError, AttributeError):
            pass
    if width == len(names):
        return list(names)
    return [f"{'_'.join(names)}_{position}" for position in range(width)]


def floats_of(values, where):
    try:
        return numpy.asarray(values, dtype="float32")
    except (ValueError, TypeError) as error:
        raise BuildError(f"{where}: the values are not numeric after the chain ({error}); add one_hot, label or "
                         f"cast") from error


def tensor_of(array, dtype):
    out = numpy.ascontiguousarray(array, dtype=dtype)
    if not out.flags.writeable:
        out = out.copy()
    return torch.from_numpy(out)


def step_names(chain):
    return [type(step).__name__ for step in chain]


def inverts(step):
    return hasattr(step, "inverse_transform") and bool(getattr(step, "rescales", True))


def label_step(chain):
    for step in chain:
        if isinstance(step, pre.Label):
            return step
    return None


def mapping_of(spec, what):
    if spec is None:
        return {}
    if not isinstance(spec, dict):
        raise BuildError(f"{what} is a mapping of column names or patterns to preprocessors, got "
                         f"{type(spec).__name__}")
    return dict(spec)


def names_of(spec, what):
    if spec is None:
        return []
    if isinstance(spec, str):
        return [spec]
    if isinstance(spec, (list, tuple)) and all(isinstance(item, str) for item in spec):
        return list(spec)
    raise BuildError(f"{what} is a column name or a list of column names, got {spec!r}")


@dataclass
class Block:
    pattern: str
    columns: list
    chain: list
    names: list = field(default_factory=list)


@dataclass
class Target:
    name: str
    chain: list
    label: object = None
    width: int = 1
    classes: list | None = None
    class_weights: object = None
