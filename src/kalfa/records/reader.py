import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy


MANIFEST = "manifest.json"
PLAN = "plan.json"
HISTORY = "history.jsonl"
EVENTS = "events.jsonl"
UPDATES = "updates.jsonl"
MODEL_SUMMARY = "model_summary.json"
FIELDS = "fitted/fields.json"
PLOTS = "plots/"
IMAGES = (".png", ".svg", ".jpg", ".jpeg", ".gif", ".pdf", ".html")
SWEEP = "sweep.json"
STALE_AFTER = 120.0
FINAL = ("done", "failed", "stopped")
HISTORY_AXES = ("turn", "update", "seconds", "rules")
UPDATE_AXES = ("update", "turn")
MINIMIZES = "minimizes/"


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_finite(value):
    return is_number(value) and math.isfinite(value)


def is_record(directory):
    return (Path(directory) / MANIFEST).is_file()


def is_sweep_root(directory):
    return (Path(directory) / SWEEP).is_file()


def read_json(directory, name):
    path = Path(directory) / name
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_lines(path):
    path = Path(path)
    if not path.is_file():
        return []
    lines = []
    with path.open(encoding="utf-8") as handle:
        for text in handle:
            if not text.strip():
                continue
            try:
                line = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(line, dict):
                lines.append(line)
    return lines


def read_manifest(directory):
    return read_json(directory, MANIFEST)


def read_plan(directory):
    return read_json(directory, PLAN)


def read_history(directory):
    return read_lines(Path(directory) / HISTORY)


def read_events(directory):
    return read_lines(Path(directory) / EVENTS)


def read_updates(directory):
    return read_lines(Path(directory) / UPDATES)


def model_summary_of(directory):
    return read_json(directory, MODEL_SUMMARY)


def data_report_of(directory):
    return read_json(directory, "data.json")


def fields_of(directory):
    try:
        fields = read_json(directory, FIELDS)
    except (OSError, ValueError):
        return None
    return fields if isinstance(fields, dict) else None


def records_under(root):
    root = Path(root)
    found = []
    gather(root, root, found)
    found.sort(key=lambda entry: entry["path"])
    found.sort(key=lambda entry: str(entry["manifest"].get("started") or ""), reverse=True)
    return found


def safe_json(directory, name):
    try:
        value = read_json(directory, name)
    except (OSError, ValueError):
        value = None
    return value if isinstance(value, dict) else {}


def gather(root, directory, found):
    if is_sweep_root(directory):
        found.append({"path": directory.relative_to(root).as_posix(), "manifest": safe_json(directory, SWEEP),
                      "kind": "sweep"})
        return
    if is_record(directory):
        found.append({"path": directory.relative_to(root).as_posix(), "manifest": safe_json(directory, MANIFEST),
                      "kind": "record"})
        return
    try:
        with os.scandir(directory) as entries:
            names = sorted(entry.name for entry in entries
                           if entry.is_dir(follow_symlinks=False) and not entry.name.startswith("."))
    except OSError:
        return
    for name in names:
        gather(root, directory / name, found)


def moment_of(value):
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, str):
        try:
            moment = datetime.fromisoformat(value.strip())
        except ValueError:
            return None
    else:
        return None
    if moment.tzinfo is not None:
        moment = moment.astimezone().replace(tzinfo=None)
    return moment


def state_of(manifest, now=None, stale_after=STALE_AFTER):
    if not isinstance(manifest, dict):
        return "unknown"
    status = manifest.get("status")
    if status in FINAL:
        return status
    if status != "running":
        return "unknown"
    updated = moment_of(manifest.get("updated"))
    if updated is None:
        return "stale"
    moment = moment_of(now) if now is not None else None
    if moment is None:
        moment = datetime.now()
    age = (moment - updated).total_seconds()
    return "stale" if age > float(stale_after) else "running"


def value_of(value):
    return value if is_finite(value) else None


def numeric_in(line, excluded):
    return {key: value_of(value) for key, value in line.items() if key not in excluded and is_number(value)}


def turn_of(line, position):
    return int(line["turn"]) if is_number(line.get("turn")) else position + 1


def normal_mode(mode):
    return "max" if str(mode or "min").lower().startswith("max") else "min"


def progress_from(manifest, history):
    manifest = manifest if isinstance(manifest, dict) else {}
    last = history[-1] if history else None
    if last is not None:
        turn = turn_of(last, len(history) - 1)
    elif is_number(manifest.get("turn")):
        turn = int(manifest["turn"])
    else:
        turn = 0
    seconds = sum((float(line["seconds"]) for line in history if is_finite(line.get("seconds"))), 0.0)
    return {"turn": turn, "total": manifest.get("turns_planned"), "status": state_of(manifest),
            "manifest_status": manifest.get("status"), "updated": manifest.get("updated"), "seconds": seconds,
            "last": numeric_in(last or {}, HISTORY_AXES), "unit": manifest.get("unit")}


def progress_of(directory):
    return progress_from(read_manifest(directory), read_history(directory))


def best_in(history, series, mode="min"):
    maximize = normal_mode(mode) == "max"
    best = None
    for position, line in enumerate(history):
        value = line.get(series)
        if not is_finite(value):
            continue
        if best is None or (value > best[0] if maximize else value < best[0]):
            best = (float(value), turn_of(line, position))
    return best if best is not None else (None, None)


def watched(plan):
    train = (plan or {}).get("train")
    checkpoint = train.get("checkpoint") if isinstance(train, dict) else None
    if isinstance(checkpoint, str):
        checkpoint = {"watch": checkpoint}
    if not isinstance(checkpoint, dict) or not checkpoint.get("watch"):
        return None, None
    return str(checkpoint["watch"]), normal_mode(checkpoint.get("mode"))


def best_from(plan, history, series=None, mode=None):
    if series is None:
        series, watched_mode = watched(plan)
        mode = mode if mode is not None else watched_mode
    if series is None:
        return None
    mode = normal_mode(mode)
    value, turn = best_in(history, series, mode)
    return {"series": str(series), "mode": mode, "value": value, "turn": turn}


def best_of(directory):
    plan = read_plan(directory)
    if watched(plan)[0] is None:
        return None
    return best_from(plan, read_history(directory))


def rules_in(history):
    fired = []
    for line in history:
        names = line.get("rules")
        if not isinstance(names, (list, tuple)):
            continue
        for name in names:
            if isinstance(name, str) and name not in fired:
                fired.append(name)
    return fired


def summary_of(directory):
    manifest = read_manifest(directory)
    manifest = manifest if isinstance(manifest, dict) else {}
    plan = read_plan(directory)
    history = read_history(directory)
    progress = progress_from(manifest, history)
    last = history[-1] if history else {}
    host = manifest.get("host")
    failure = manifest.get("failure")
    return {
        "path": str(directory),
        "name": manifest.get("name") or Path(directory).name,
        "status": progress["status"],
        "unit": progress["unit"],
        "turn": progress["turn"],
        "total": progress["total"],
        "started": manifest.get("started"),
        "updated": progress["updated"],
        "finished": manifest.get("finished"),
        "seconds": progress["seconds"],
        "params": dict(manifest.get("params") or {}),
        "best": best_from(plan, history),
        "last": progress["last"],
        "minimizes": {key[len(MINIMIZES):]: value for key, value in last.items()
                      if key.startswith(MINIMIZES) and isinstance(value, str)},
        "rules": rules_in(history),
        "device": manifest.get("device"),
        "host": host.get("hostname") if isinstance(host, dict) else host if isinstance(host, str) else None,
        "failure": failure.get("message") if isinstance(failure, dict) else str(failure) if failure else None,
        "sweep": manifest.get("sweep") if isinstance(manifest.get("sweep"), dict) else None,
        "resumed_from": manifest.get("resumed_from"),
        "stop_reason": manifest.get("stop_reason"),
        "stopped_by": manifest.get("stopped_by"),
    }


def numeric_keys(lines, excluded):
    keys = []
    for line in lines:
        for key, value in line.items():
            if key not in excluded and key not in keys and is_number(value):
                keys.append(key)
    return keys


def thinned(count, limit):
    if not limit or limit < 1 or count <= limit:
        return range(count)
    positions = numpy.unique(numpy.rint(numpy.linspace(0, count - 1, num=int(limit))).astype(int))
    return [int(position) for position in positions]


def history_axis(line, position, axis):
    return turn_of(line, position)


def update_axis(line, position, axis):
    return value_of(line.get(axis))


def table_of(lines, axes, axis_of, excluded, keys, limit):
    if isinstance(keys, str):
        keys = [keys]
    keys = numeric_keys(lines, excluded) if keys is None else [str(key) for key in keys]
    positions = thinned(len(lines), limit)
    table = {axis: [] for axis in axes}
    for key in keys:
        table.setdefault(key, [])
    for position in positions:
        line = lines[position]
        for axis in axes:
            table[axis].append(axis_of(line, position, axis))
        for key in keys:
            table[key].append(value_of(line.get(key)))
    return table


def series_of(directory, keys=None, limit=2000):
    return table_of(read_history(directory), ("turn",), history_axis, HISTORY_AXES, keys, limit)


def updates_of(directory, keys=None, limit=2000):
    return table_of(read_updates(directory), UPDATE_AXES, update_axis, UPDATE_AXES, keys, limit)


def events_of(directory, kinds=None, limit=None):
    events = read_events(directory)
    if kinds is not None:
        wanted = {kinds} if isinstance(kinds, str) else set(kinds)
        events = [event for event in events if event.get("kind") in wanted]
    if limit:
        events = events[-int(limit):]
    return events


def files_of(directory):
    root = Path(directory)
    if not root.is_dir():
        return []
    found = []
    for current, names, files in os.walk(root):
        names[:] = sorted(name for name in names if not name.startswith("."))
        for name in files:
            if name.startswith("."):
                continue
            path = Path(current) / name
            try:
                info = path.stat()
            except OSError:
                continue
            found.append({"path": path.relative_to(root).as_posix(), "size": int(info.st_size),
                          "modified": datetime.fromtimestamp(info.st_mtime).isoformat(timespec="seconds")})
    found.sort(key=lambda entry: entry["path"])
    return found


def plots_of(directory):
    found = []
    for entry in files_of(directory):
        path = entry["path"]
        if path.startswith(PLOTS) and path.lower().endswith(IMAGES):
            found.append({"name": Path(path[len(PLOTS):]).with_suffix("").as_posix(), **entry})
    return found
