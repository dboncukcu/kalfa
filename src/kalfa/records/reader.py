import json
import math
import os
import threading
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
LOG = "log.txt"
STALE_AFTER = 120.0
RECENT_TURNS = 5
AFTER = ("after", "predictions", "samples", "plots")
BLOCK = 65_536
MISSING = "-"
STAMPED = (MANIFEST, HISTORY, EVENTS, UPDATES)
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


def within_of(manifest, turn):
    within = manifest.get("within") if isinstance(manifest, dict) else None
    if not isinstance(within, dict) or manifest.get("status") != "running":
        return None
    if not is_number(within.get("turn")):
        return None
    current, phase = int(within["turn"]), str(within.get("phase") or "train")
    if current != turn + 1 and not (current == turn and phase in AFTER):
        return None
    done, total = within.get("done"), within.get("total")
    return {"turn": current, "phase": phase, "done": int(done) if is_finite(done) else None,
            "total": int(total) if is_finite(total) else None, "unit": within.get("unit"),
            "started": within.get("started")}


def share_of(within):
    if within is None or within["phase"] in AFTER:
        return 0.0
    if within["phase"] != "train":
        return 1.0
    done, total = within["done"], within["total"]
    if done is None or not total or total <= 0:
        return 0.0
    return max(0.0, min(1.0, done / total))


def fraction_of(turn, total, share):
    if not is_finite(total) or total <= 0:
        return None
    return max(0.0, min(1.0, (turn + share) / total))


def turn_seconds(manifest, history, within, share):
    recent = [float(line["seconds"]) for line in history[-RECENT_TURNS:] if is_finite(line.get("seconds"))]
    if recent:
        return sum(recent) / len(recent)
    if within is None or share <= 0:
        return None
    started, updated = moment_of(within.get("started")), moment_of(manifest.get("updated"))
    if started is None or updated is None or updated <= started:
        return None
    return (updated - started).total_seconds() / share


def eta_of(manifest, history, turn, within, share):
    total = manifest.get("turns_planned")
    if manifest.get("status") != "running" or not is_finite(total) or total <= 0:
        return None
    each = turn_seconds(manifest, history, within, share)
    if each is None:
        return None
    return max(0.0, each * (total - turn - share))


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
    within = within_of(manifest, turn)
    share = share_of(within)
    return {"turn": turn, "total": manifest.get("turns_planned"), "status": state_of(manifest),
            "manifest_status": manifest.get("status"), "updated": manifest.get("updated"), "seconds": seconds,
            "last": numeric_in(last or {}, HISTORY_AXES), "unit": manifest.get("unit"), "within": within,
            "fraction": fraction_of(turn, manifest.get("turns_planned"), share),
            "eta": eta_of(manifest, history, turn, within, share)}


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


def rule_turns(history):
    found = []
    for position, line in enumerate(history):
        names = line.get("rules")
        if isinstance(names, (list, tuple)) and names:
            found.append({"turn": turn_of(line, position), "update": value_of(line.get("update")),
                          "rules": [str(name) for name in names]})
    return found


def number_word(value):
    if math.isnan(value):
        return "nan"
    return "inf" if value > 0 else "-inf"


def nonfinite_in(history):
    for position, line in enumerate(history):
        bad = [key for key, value in line.items()
               if key not in HISTORY_AXES and is_number(value) and not math.isfinite(value)]
        if bad:
            return {"turn": turn_of(line, position), "series": bad, "value": number_word(line[bad[0]])}
    return None


def failure_of(directory):
    try:
        manifest = read_manifest(directory)
    except (OSError, ValueError):
        return None
    failure = manifest.get("failure") if isinstance(manifest, dict) else None
    if isinstance(failure, dict):
        return {"type": failure.get("type"), "message": failure.get("message"),
                "traceback": failure.get("traceback")}
    return {"type": None, "message": str(failure), "traceback": None} if failure else None


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
        "within": progress["within"],
        "fraction": progress["fraction"],
        "eta": progress["eta"],
        "nonfinite": nonfinite_in(history),
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


def last_lines(path, count):
    if count <= 0:
        return b""
    with Path(path).open("rb") as handle:
        position = handle.seek(0, os.SEEK_END)
        blocks = []
        newlines = 0
        while position > 0 and newlines <= count:
            step = min(BLOCK, position)
            position -= step
            handle.seek(position)
            block = handle.read(step)
            newlines += block.count(b"\n")
            blocks.append(block)
    data = b"".join(reversed(blocks))
    return b"".join(data.splitlines(keepends=True)[-count:])


def log_of(directory, count=200):
    path = Path(directory) / LOG
    try:
        data = last_lines(path, count) if path.is_file() else b""
    except OSError:
        return []
    return data.decode("utf-8", errors="replace").splitlines()


def file_stamp(path):
    try:
        info = Path(path).stat()
    except OSError:
        return MISSING
    return f"{info.st_mtime_ns}:{info.st_size}"


def record_stamp(directory, manifest):
    directory = Path(directory)
    manifest = manifest if isinstance(manifest, dict) else {}
    files = "|".join(file_stamp(directory / name) for name in STAMPED)
    return f"{manifest.get('updated')}|{state_of(manifest)}|{files}"


class Cache:
    def __init__(self, limit=8192):
        self.limit = int(limit)
        self.kept = {}
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key, stamp, compute):
        with self.lock:
            found = self.kept.get(key)
            if found is not None and found[0] == stamp:
                self.hits += 1
                return found[1]
            self.misses += 1
        value = compute()
        with self.lock:
            if len(self.kept) >= self.limit:
                self.kept.clear()
            self.kept[key] = (stamp, value)
        return value


def summary_with(cache, directory, manifest=None):
    if cache is None:
        return summary_of(directory)
    if manifest is None:
        manifest = safe_json(directory, MANIFEST)
    return cache.get(("summary", str(directory)), record_stamp(directory, manifest), lambda: summary_of(directory))
