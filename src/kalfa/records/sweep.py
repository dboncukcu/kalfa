import json
import math
import re
from pathlib import Path

import numpy

from .reader import (
    MANIFEST,
    SWEEP,
    best_in,
    is_finite,
    is_number,
    is_sweep_root,
    normal_mode,
    read_history,
    read_json,
    record_stamp,
    safe_json,
    summary_of,
)


POINT_NAME = re.compile(r"^\d{4}$")
FAILED = "failed"


is_sweep = is_sweep_root


def read_sweep(root):
    return read_json(root, SWEEP)


def point_dirs(root):
    root = Path(root)
    try:
        found = [entry for entry in root.iterdir() if entry.is_dir() and POINT_NAME.match(entry.name)]
    except OSError:
        return []
    return sorted(found, key=lambda path: int(path.name))


def objective_of(summary, objective):
    if not isinstance(summary, dict) or not isinstance(objective, dict):
        return None
    watch = objective.get("watch")
    if watch is None:
        return None
    if str(objective.get("at") or "best") == "last":
        last = summary.get("last")
        value = last.get(watch) if isinstance(last, dict) else None
    else:
        value = best_value(summary, watch, mode_of(objective))
    return float(value) if is_finite(value) else None


def best_value(summary, watch, mode):
    best = summary.get("best")
    if isinstance(best, dict) and best.get("series") == watch and normal_mode(best.get("mode")) == mode:
        return best.get("value")
    path = summary.get("path")
    if not isinstance(path, (str, Path)):
        return None
    return best_in(read_history(path), watch, mode)[0]


def point_id(summary, directory):
    sweep = summary.get("sweep")
    if isinstance(sweep, dict) and is_number(sweep.get("id")):
        return int(sweep["id"])
    return int(directory.name)


def point_values(summary):
    sweep = summary.get("sweep")
    if isinstance(sweep, dict) and isinstance(sweep.get("values"), dict):
        return dict(sweep["values"])
    params = summary.get("params")
    return dict(params) if isinstance(params, dict) else {}


def point_of(directory, root, objective):
    summary = summary_of(directory)
    if not isinstance(summary, dict):
        return None
    status = summary.get("status")
    return {
        "id": point_id(summary, directory),
        "path": directory.relative_to(root).as_posix(),
        "values": point_values(summary),
        "status": status,
        "turn": summary.get("turn"),
        "total": summary.get("total"),
        "objective": None if status == FAILED else objective_of(summary, objective),
        "summary": summary,
    }


def sweep_points(root, cache=None):
    root = Path(root)
    sweep = read_sweep(root)
    objective = sweep.get("objective") if isinstance(sweep, dict) else None
    asked = json.dumps(objective, sort_keys=True, default=str)
    points = []
    for directory in point_dirs(root):
        if cache is None:
            point = point_of(directory, root, objective)
        else:
            stamp = record_stamp(directory, safe_json(directory, MANIFEST))
            point = cache.get(("point", str(directory), str(root), asked), stamp,
                              lambda directory=directory: point_of(directory, root, objective))
        if point is not None:
            points.append(dict(point))
    return points


def is_valued(point):
    return is_finite(point.get("objective")) and point.get("status") != FAILED


def id_key(point):
    identity = point.get("id")
    return (0, identity, "") if is_number(identity) else (1, 0, str(identity))


def mode_of(objective):
    return normal_mode(objective.get("mode") if isinstance(objective, dict) else objective)


def rank_points(points, objective):
    maximize = mode_of(objective) == "max"
    valued = [point for point in points if is_valued(point)]
    unvalued = [point for point in points if not is_valued(point)]
    valued.sort(key=lambda point: ((-point["objective"] if maximize else point["objective"]), id_key(point)))
    unvalued.sort(key=id_key)
    return valued + unvalued


def best_point(points, objective):
    ranked = rank_points(points, objective)
    if ranked and is_valued(ranked[0]):
        return ranked[0]
    return None


def running_best(points, objective):
    maximize = mode_of(objective) == "max"
    ordered = sorted(points, key=id_key)
    best = None
    found = {"ids": [], "objective": [], "best": []}
    for point in ordered:
        value = point["objective"] if is_valued(point) else None
        if value is not None and (best is None or (value > best if maximize else value < best)):
            best = value
        found["ids"].append(point.get("id"))
        found["objective"].append(value)
        found["best"].append(best)
    return found


def is_log_space(spec):
    return isinstance(spec, dict) and isinstance(spec.get("log"), (list, tuple))


def label_of(value):
    if isinstance(value, float):
        return f"{value:.4g}"
    return value if isinstance(value, str) else json.dumps(value, default=str)


def axis_of(key, values, spec):
    numeric = all(is_finite(value) for value in values)
    if numeric and is_log_space(spec) and all(value > 0 for value in values):
        shown = sorted(set(values))
        return {"key": key, "label": f"{key} (log)", "values": [math.log10(value) for value in values],
                "ticks": [math.log10(value) for value in shown] if len(shown) <= 12 else None,
                "text": [label_of(float(value)) for value in shown] if len(shown) <= 12 else None}
    if numeric:
        return {"key": key, "label": key, "values": [float(value) for value in values], "ticks": None, "text": None}
    names = []
    for value in values:
        name = label_of(value)
        if name not in names:
            names.append(name)
    return {"key": key, "label": key, "values": [float(names.index(label_of(value))) for value in values],
            "ticks": [float(position) for position in range(len(names))], "text": names}


def axes_of(points, sweep):
    sweep = sweep if isinstance(sweep, dict) else {}
    objective = sweep.get("objective") if isinstance(sweep.get("objective"), dict) else {}
    space = sweep.get("space") if isinstance(sweep.get("space"), dict) else {}
    valued = sorted((point for point in points if is_valued(point)), key=id_key)
    keys = []
    for point in valued:
        keys.extend(key for key in point["values"] if key not in keys)
    keys = [key for key in keys if all(key in point["values"] for point in valued)]
    axes = [axis_of(key, [point["values"][key] for point in valued], space.get(key)) for key in keys]
    watch = str(objective.get("watch") or "objective")
    axes.append({"key": "objective", "label": watch, "values": [float(point["objective"]) for point in valued],
                 "ticks": None, "text": None})
    return {"ids": [point.get("id") for point in valued], "axes": axes if valued else [],
            "left_out": len(points) - len(valued), "mode": mode_of(objective)}


def frozen(value):
    if isinstance(value, dict):
        return tuple(sorted((str(key), frozen(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple, set)):
        return tuple(frozen(item) for item in value)
    return value


def numeric_last(summary):
    last = summary.get("last")
    if not isinstance(last, dict):
        return {}
    return {key: float(value) for key, value in last.items() if is_finite(value)}


def group_folds(summaries, key="fold"):
    groups = {}
    for summary in summaries:
        if not isinstance(summary, dict):
            continue
        params = summary.get("params")
        if not isinstance(params, dict) or key not in params:
            continue
        rest = {name: value for name, value in params.items() if name != key}
        group_key = frozen(rest)
        if group_key not in groups:
            groups[group_key] = {"group": rest, "members": [], "mean": {}, "std": {}}
        groups[group_key]["members"].append(summary)
    for group in groups.values():
        lasts = [numeric_last(member) for member in group["members"]]
        common = set(lasts[0])
        for last in lasts[1:]:
            common &= set(last)
        for series in sorted(common):
            values = numpy.asarray([last[series] for last in lasts], dtype=float)
            group["mean"][series] = float(values.mean())
            group["std"][series] = float(numpy.std(values))
    return list(groups.values())
