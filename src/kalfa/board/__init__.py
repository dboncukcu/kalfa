import ast
import difflib
import json
import math
import mimetypes
import os
import re
import threading
import time
from collections import OrderedDict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy
import pandas

from kalfa import api
from kalfa.errors import KalfaError
from kalfa.record import Record, failure_text, read_resolved
from kalfa.std.common.files import read_json, read_lines
from kalfa.std.common.history import History
from kalfa.std.common.log import logger_for
from kalfa.std.plot.base import prediction_pairs, r2_of, shared_histograms, true_column
from kalfa.std.pre.base import Grouped, read_prep


logger = logger_for("board")

TEXT_SUFFIXES = (".yaml", ".yml", ".json", ".jsonl", ".txt", ".md", ".csv", ".tsv", ".py", ".sh", ".sub", ".plan",
                 ".log", ".toml", ".ini", ".cfg", ".rst", ".html", ".xml")
STATIC = (Path(__file__).parent / "static").resolve()


def relative_to(root, path):
    return str(Path(path).resolve().relative_to(root))


def last_line(path):
    if not path.is_file():
        return None
    with open(path, "rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - 8192))
        tail = stream.read().decode("utf-8", errors="replace")
    for text in reversed(tail.splitlines()):
        if text.strip():
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                continue
    return None


def turns_planned(training):
    if training.get("steps") is not None:
        steps = training.get("steps") or {}
        total, turn = steps.get("total"), steps.get("turn")
        counted = isinstance(total, (int, float)) and isinstance(turn, (int, float)) and turn
        return math.ceil(total / turn) if counted else None
    epochs = training.get("epochs")
    return int(epochs) if isinstance(epochs, (int, float)) else None


def monitor_key(training):
    checkpoint = training.get("checkpoint")
    if isinstance(checkpoint, dict):
        return (checkpoint.get("params") or {}).get("monitor")
    return None


def is_number(value):
    return isinstance(value, (int, float, numpy.integer, numpy.floating)) and not isinstance(value, bool)


def clean(value):
    if isinstance(value, numpy.generic):
        value = value.item()
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


def axis_of(declared, seen):
    text = declared if isinstance(declared, str) else ""
    numbers = [value for value in seen if is_number(value)]
    choices = re.search(r"Choices\(values=(\[[^\]]*\])", text)
    if choices:
        try:
            return {"kind": "choices", "values": list(ast.literal_eval(choices.group(1)))}
        except (ValueError, SyntaxError):
            pass
    bounds = re.search(r"Range\(([^)]*)\)", text)
    if bounds:
        body = bounds.group(1)
        low = re.search(r"\blow=([-\d.eE+]+)", body)
        high = re.search(r"\bhigh=([-\d.eE+]+)", body)
        return {"kind": "range", "log": "log=True" in body,
                "low": float(low.group(1)) if low else (min(numbers) if numbers else None),
                "high": float(high.group(1)) if high else (max(numbers) if numbers else None)}
    unique = sorted(set(numbers))
    if numbers and len(unique) <= 6:
        return {"kind": "choices", "values": unique}
    if numbers:
        low, high = min(numbers), max(numbers)
        return {"kind": "range", "log": low > 0 and high / low >= 100, "low": low, "high": high}
    return {"kind": "choices", "values": sorted({str(value) for value in seen})}


def unit_of(manifest):
    return "epoch" if (manifest or {}).get("turn") == "epoch" else "turn"


def sweep_unit(points):
    return "epoch" if points and all(point["unit"] == "epoch" for point in points) else "turn"


def sweep_state(states):
    if "running" in states:
        return "running"
    if "pending" in states:
        return "pending"
    if any(state in ("failed", "lost", "unreadable") for state in states):
        return "failed"
    return "finished" if states else "pending"


def queued(states, total):
    planned = int(total) if isinstance(total, (int, float)) and not isinstance(total, bool) else 0
    return [*states, *["pending"] * max(planned - len(states), 0)]


def settle(entries):
    below = {}
    for entry in entries:
        below.setdefault(str(Path(entry["path"]).parent), []).append(entry["status"]["state"])
    for entry in entries:
        if entry["kind"] != "sweep":
            continue
        states = queued(below.get(entry["path"], []), entry.get("total"))
        if states:
            entry["status"] = {**entry["status"], "state": sweep_state(states)}
    return entries


def finite_or_text(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def marked(line):
    return {key: finite_or_text(value) for key, value in line.items()} if isinstance(line, dict) else line


def unreadable_point(relative, problem):
    return {"path": relative, "id": None, "values": {}, "status": {"state": "unreadable", "error": str(problem)},
            "objective": None, "turns": 0, "unit": "turn", "started": None, "error": str(problem)}


QUERY_NODES = (ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub, ast.UAdd, ast.Invert,
               ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.BitAnd, ast.BitOr,
               ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Name,
               ast.Load, ast.Constant, ast.List, ast.Tuple)


def checked_where(text, columns):
    quoted = {}

    def hold(match):
        name = f"_quoted_{len(quoted)}"
        quoted[name] = match.group(1)
        return name

    try:
        tree = ast.parse(re.sub(r"`([^`]*)`", hold, text), mode="eval")
    except SyntaxError as problem:
        raise ValueError(f"the filter is no expression ({problem.msg})") from None
    for node in ast.walk(tree):
        if not isinstance(node, QUERY_NODES):
            raise ValueError(f"the filter takes column names, numbers, strings, comparisons, and, or, not, "
                             f"arithmetic and lists; {type(node).__name__} is not one of them")
        if isinstance(node, ast.Name) and quoted.get(node.id, node.id) not in columns:
            raise ValueError(f"{quoted.get(node.id, node.id)} is no column of the predictions")
    return text


def filtered(table, where):
    if not where:
        return table, None
    try:
        return table.query(checked_where(where, set(table.columns))), None
    except Exception as problem:
        return table.iloc[:0], f"{type(problem).__name__}: {problem}"


EFFICIENCIES = (0.5, 0.8, 0.9, 0.95)


def roc_of(values, signal, limit=400):
    positives, negatives = int(signal.sum()), int((~signal).sum())
    if not positives or not negatives:
        return None
    order = numpy.argsort(-values, kind="mergesort")
    ranked, hits = values[order], signal[order]
    ends = numpy.r_[numpy.flatnonzero(numpy.diff(ranked)), len(ranked) - 1]
    tpr = numpy.r_[0.0, numpy.cumsum(hits)[ends] / positives]
    fpr = numpy.r_[0.0, numpy.cumsum(~hits)[ends] / negatives]
    cuts = numpy.r_[numpy.inf, ranked[ends]]
    working = []
    for efficiency in EFFICIENCIES:
        at = int(numpy.searchsorted(tpr, efficiency))
        if at < len(tpr):
            background = float(fpr[at])
            working.append({"signal": float(tpr[at]), "background": background,
                            "rejection": 1 / background if background > 0 else None, "cut": float(cuts[at])})
    shown = numpy.unique(numpy.linspace(0, len(tpr) - 1, min(limit, len(tpr))).round().astype(int))
    return {"auc": float(numpy.sum(numpy.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2)), "fpr": fpr[shown].tolist(),
            "tpr": tpr[shown].tolist(), "working": working, "signal": positives, "background": negatives}


def class_histograms(values, texts, classes, bins=40):
    if not len(values):
        return None
    low, high = float(values.min()), float(values.max())
    edges = numpy.linspace(low, high if high > low else low + 1.0, max(2, int(bins)) + 1)
    return {"edges": edges.tolist(),
            "classes": [{"label": item, "counts": numpy.histogram(values[texts == str(item)], bins=edges)[0].tolist()}
                        for item in classes]}


def read_space(manifest, points):
    declared = (manifest or {}).get("space") or {}
    keys = list(declared) or sorted({key for point in points for key in point.get("values") or {}})
    return {key: axis_of(declared.get(key), [point["values"][key] for point in points
                                             if key in (point.get("values") or {})]) for key in keys}


def small_value(value, depth=0):
    if isinstance(value, numpy.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) else round(value, 6)
    if isinstance(value, numpy.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        items = [small_value(item, depth + 1) for item in value[:512]]
        if any(item is None for item in items) and not all(item is None for item in items):
            return None
        return items if len(value) <= 512 else [*items, f"and {len(value) - 512} more"]
    if isinstance(value, dict):
        if len(value) > 64:
            return f"{len(value)} entries"
        return {str(key): small_value(item, depth + 1) for key, item in value.items()}
    if depth < 2 and hasattr(value, "__dict__"):
        return small_state(value, depth + 1)
    return type(value).__name__


def small_state(obj, depth=0):
    state = {}
    for name, value in vars(obj).items():
        if name.startswith("_"):
            continue
        kept = small_value(value, depth)
        if kept is not None:
            state[name] = kept
    return state


def stamp(path):
    try:
        stat = path.stat()
    except OSError:
        return None
    return [stat.st_size, stat.st_mtime_ns]


def stamp_dir(path):
    if not path.is_dir():
        return None
    files = [item for item in path.iterdir() if item.is_file()]
    return [len(files), max((item.stat().st_mtime_ns for item in files), default=0)]


def file_note(root, path):
    stat = path.stat()
    return {"name": str(path.relative_to(root)), "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")}


def static_path(name):
    target = (STATIC / (name or "")).resolve()
    if not target.is_relative_to(STATIC) or not target.is_file():
        return None
    return target


def resolved_at(path):
    return read_resolved(path.parent)


def best_point(points, objective):
    scored = [point for point in points if point["objective"] is not None
              and (point.get("status") or {}).get("state") not in ("failed", "lost", "unreadable")]
    pick = min if objective.get("mode", "min") == "min" else max
    return pick(scored, key=lambda point: point["objective"]["value"]) if scored else None


class Lines:
    def __init__(self, room=64):
        self.room = room
        self.kept = OrderedDict()
        self.lock = threading.Lock()

    def read(self, path):
        with self.lock:
            try:
                found = os.stat(path)
            except OSError:
                self.kept.pop(path, None)
                return []
            kept = self.kept.pop(path, None)
            if kept is None or kept["ino"] != found.st_ino or found.st_size < kept["consumed"]:
                kept = {"ino": found.st_ino, "consumed": 0, "lines": [], "broken": False}
            if found.st_size > kept["consumed"] and not kept["broken"]:
                with open(path, "rb") as stream:
                    stream.seek(kept["consumed"])
                    chunk = stream.read(found.st_size - kept["consumed"])
                end = chunk.rfind(b"\n")
                if end >= 0:
                    for raw in chunk[:end].split(b"\n"):
                        text = raw.decode("utf-8", errors="replace")
                        if not text.strip():
                            continue
                        try:
                            kept["lines"].append(json.loads(text))
                        except json.JSONDecodeError:
                            kept["broken"] = True
                            break
                    kept["consumed"] += end + 1
            self.kept[path] = kept
            while len(self.kept) > self.room:
                self.kept.popitem(last=False)
            return list(kept["lines"])


class Counts:
    def __init__(self):
        self.kept = {}
        self.lock = threading.Lock()

    def total(self, path):
        with self.lock:
            found = os.stat(path)
            ino, consumed, breaks, last = self.kept.get(path, (None, 0, 0, b""))
            if ino != found.st_ino or found.st_size < consumed:
                ino, consumed, breaks, last = found.st_ino, 0, 0, b""
            if found.st_size > consumed:
                with open(path, "rb") as stream:
                    stream.seek(consumed)
                    while chunk := stream.read(1 << 20):
                        breaks += chunk.count(b"\n")
                        consumed += len(chunk)
                        last = chunk[-1:]
            self.kept[path] = (ino, consumed, breaks, last)
            return breaks + (1 if consumed and last != b"\n" else 0)


def tail_lines(path, count, window=1 << 19):
    size = path.stat().st_size
    while True:
        with path.open("rb") as stream:
            stream.seek(max(0, size - window))
            text = stream.read(window).decode("utf-8", errors="replace").split("\n")
        if window >= size or len(text) > count + 1:
            break
        window *= 4
    if size > window:
        text = text[1:]
    if text and text[-1] == "":
        text.pop()
    return [line.rstrip("\r").rsplit("\r", 1)[-1] for line in text[-count:]]


class Watcher:
    def __init__(self, board, every=2.0, idle=30.0):
        self.board = board
        self.every = every
        self.idle = idle
        self.lock = threading.Lock()
        self.common = None
        self.viewers = 0
        self.alone = 0.0
        self.thread = None

    def join(self):
        with self.lock:
            self.viewers += 1
            if self.thread is None:
                self.thread = threading.Thread(target=self.loop, name="kalfa-board-watch", daemon=True)
                self.thread.start()

    def leave(self):
        with self.lock:
            self.viewers -= 1
            if not self.viewers:
                self.alone = time.monotonic()

    def current(self):
        with self.lock:
            return self.common

    def loop(self):
        while True:
            try:
                common = self.board.common_stamps()
            except Exception as problem:
                logger.warning(f"watch: {type(problem).__name__}: {problem}")
                common = None
            with self.lock:
                if common is not None:
                    self.common = common
                if not self.viewers and time.monotonic() - self.alone > self.idle:
                    self.thread = None
                    self.common = None
                    return
            time.sleep(self.every)


class Cache:
    def __init__(self):
        self.kept = {}

    def read(self, path, parse):
        try:
            found = os.stat(path)
        except OSError:
            return None
        key = (found.st_ino, found.st_size, found.st_mtime_ns)
        kept = self.kept.get((path, parse))
        if kept is not None and kept[0] == key:
            return kept[1]
        value = parse(path)
        self.kept[(path, parse)] = (key, value)
        return value

    def json(self, path):
        return self.read(path, read_json)


class Frames:
    def __init__(self, room=4, budget=1 << 30):
        self.room = room
        self.budget = budget
        self.kept = OrderedDict()
        self.lock = threading.Lock()

    def read(self, path):
        found = os.stat(path)
        key = (found.st_ino, found.st_size, found.st_mtime_ns)
        with self.lock:
            kept = self.kept.get(path)
            if kept is not None and kept[0] == key:
                self.kept.move_to_end(path)
                return kept[1]
        table = pandas.read_parquet(path)
        with self.lock:
            self.kept[path] = (key, table, int(table.memory_usage(index=True).sum()))
            self.kept.move_to_end(path)
            while self.kept and (len(self.kept) > self.room
                                 or sum(item[2] for item in self.kept.values()) > self.budget):
                self.kept.popitem(last=False)
        return table


class Board:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.cache = Cache()
        self.frames = Frames()
        self.jsonl = Lines()
        self.counts = Counts()
        self.watcher = Watcher(self)

    def history(self, path):
        return History(self.cache.read(path / "history.jsonl", read_lines))

    def config(self, path):
        return self.cache.read(path / "resolved.yaml", resolved_at)

    def resolve(self, relative):
        target = (self.root / (relative or "")).resolve()
        if target != self.root and self.root not in target.parents:
            return None
        return target

    def records(self, full=True):
        found = []
        self.gather(self.root, found, full)
        return settle(found)

    def gather(self, directory, found, full, inside=False):
        try:
            with os.scandir(directory) as entries:
                names = sorted(entry.name for entry in entries if entry.is_dir(follow_symlinks=False))
        except OSError:
            return
        for name in names:
            path = directory / name
            record = Record(path, self.cache.json)
            if not record.is_record:
                if not inside:
                    self.gather(path, found, full)
                continue
            try:
                manifest = record.read_json("manifest.json") or {}
                status = record.status() if full else {"state": record.state()}
            except (OSError, ValueError) as problem:
                logger.warning(f"{path}: unreadable ({problem})")
                manifest, status = {}, {"state": "unreadable", "error": str(problem)}
            kind = manifest.get("kind") or "run"
            if kind == "data":
                continue
            found.append({"path": str(path.relative_to(self.root)), "kind": kind, "name": manifest.get("name") or name,
                          "started": manifest.get("started"), "status": status, "unit": unit_of(manifest),
                          **({"total": manifest.get("total")} if kind == "sweep" else {})})
            if kind == "sweep":
                self.gather(path, found, full, inside=True)

    def series(self, relatives, keys=None, limit=200):
        runs, names = {}, set()
        for relative in relatives[:limit]:
            path = self.resolve(relative)
            if path is None or not Record(path).is_record:
                continue
            history = self.history(path)
            manifest = self.cache.json(path / "manifest.json") or {}
            present = {key for line in history for key, value in line.items()
                       if key not in ("turn", "global_step", "seconds") and is_number(value)}
            names |= present
            wanted = sorted(present) if keys is None else [key for key in keys if key in present]
            runs[relative] = {"name": manifest.get("name") or path.name, "unit": unit_of(manifest),
                              "params": manifest.get("params") or {},
                              "series": {key: [[line.get("turn"), finite_or_text(line[key])] for line in history
                                               if is_number(line.get(key))] for key in wanted}}
        missing = [item for item in relatives[:limit] if item not in runs]
        return {"runs": runs, "keys": sorted(names), "missing": missing}

    def tree(self):
        groups = {}
        for entry in self.records():
            parent = str(Path(entry["path"]).parent)
            groups.setdefault(parent, []).append(entry)
        return {"root": str(self.root), "groups": groups}

    def best_of(self, config, history):
        training = (config or {}).get("training") or {}
        monitor = monitor_key(training)
        checkpoint = training.get("checkpoint")
        mode = ((checkpoint.get("params") or {}).get("mode") or "min") if isinstance(checkpoint, dict) else "min"
        if not monitor or not len(history):
            return None
        try:
            value, turn = history.best(monitor, mode)
        except ValueError:
            return None
        return {"monitor": monitor, "mode": mode, "value": value, "turn": turn}

    def record(self, relative):
        path = self.resolve(relative)
        if path is None or not Record(path).is_record:
            return None
        record = Record(path, self.cache.json)
        plots = sorted(item.name for item in (path / "plots").glob("*") if item.is_file())
        samples = sorted(item.name for item in (path / "samples").glob("turn_*.png"))
        resolved = path / "resolved.yaml"
        config = self.config(path)
        checkpoints = [file_note(path, item) for folder in ("checkpoints", "final", "export")
                       for item in sorted((path / folder).glob("*")) if item.is_file()]
        return {"path": relative, "status": record.status(), "manifest": record.read_json("manifest.json"),
                "best": self.best_of(config, self.history(path)), "checkpoints": checkpoints,
                "calibrations": self.cache.json(path / "fitted" / "calibrate" / "calibrate.json"),
                "predictions": sorted(item.name for item in path.glob("predictions*.parquet")),
                "host": record.read_json("host.json"), "device": record.read_json("device.json"),
                "git": record.read_json("git.json"), "resume": record.read_json("resume.json"),
                "sweep": record.read_json("sweep.json"), "data": record.read_json("data.json"),
                "run": record.read_json("run.json"), "architecture": record.read_json("architecture.json"),
                "failure": record.read_json("failure.json"), "repair": record.read_json("repair.json"),
                "plots": plots, "samples": samples,
                "resolved": resolved.read_text() if resolved.exists() else None,
                "config": config,
                "events": self.jsonl.read(path / "events.jsonl")[-60:],
                "logs": [name for name in ("stdout.txt", "stderr.txt") if (path / name).is_file()]}

    def progress(self, path):
        config = self.config(path)
        training = (config or {}).get("training") or {}
        history = self.history(path)
        last = history[-1] if len(history) else {}
        seconds = [line["seconds"] for line in history if isinstance(line.get("seconds"), (int, float))]
        planned = turns_planned(training)
        remaining = planned - len(history) if planned is not None else None
        eta = sum(seconds) / len(seconds) * remaining if seconds and remaining is not None and remaining > 0 else None
        step = last_line(path / "steps.jsonl") or {}
        report = self.cache.json(path / "data.json") or {}
        batches = ((report.get("loaders") or {}).get("train") or {}).get("batches")
        monitor = monitor_key(training)
        shown = {key: finite_or_text(value) for key, value in last.items()
                 if key.startswith(("val/", "test/")) and isinstance(value, (int, float)) and "/total" not in key}
        current = step.get("step")
        return {"turn": len(history), "turns_total": planned, "eta": eta, "monitor": monitor,
                "monitor_value": finite_or_text(last.get(monitor)) if monitor else None, "step": current,
                "step_in_turn": current - last.get("global_step", 0) if current is not None else None,
                "steps_per_turn": batches,
                "losses": {key: finite_or_text(value) for key, value in step.items() if key.startswith("loss/")},
                "last": dict(list(shown.items())[:8])}

    def live(self):
        entries = self.records()
        below = {}
        for entry in entries:
            if entry["kind"] != "sweep":
                below.setdefault(str(Path(entry["path"]).parent), []).append(entry)
        live, recent = [], []
        for entry in entries:
            if entry["kind"] == "sweep":
                points = below.get(entry["path"], [])
                states = queued([point["status"]["state"] for point in points], entry.get("total"))
                active = any(state in ("running", "pending") for state in states)
                manifest, objective, best = self.standing(entry["path"], points)
                note = {**entry, "finished": states.count("finished"), "running": states.count("running"),
                        "failed": states.count("failed"), "lost": states.count("lost"),
                        "total": manifest.get("total") or len(states), "objective": objective, "best": best}
                (live if active else recent).append(note)
                continue
            if entry["status"]["state"] in ("running", "pending"):
                live.append({**entry, **self.progress(self.root / entry["path"])})
            else:
                recent.append(entry)
        recent.sort(key=lambda item: item["status"].get("last_seen") or "", reverse=True)
        return {"live": live, "recent": recent[:12]}

    def table(self):
        rows = []
        for entry in self.records():
            if entry["kind"] == "sweep":
                continue
            path = self.root / entry["path"]
            record = Record(path, self.cache.json)
            config = self.config(path)
            history = self.history(path)
            last = history[-1] if len(history) else {}
            manifest = record.read_json("manifest.json") or {}
            rows.append({**entry, "params": manifest.get("params") or {}, "turns": len(history),
                         "seconds": sum(line["seconds"] for line in history if is_number(line.get("seconds"))),
                         "best": self.best_of(config, history),
                         "device": (record.read_json("device.json") or {}).get("device"),
                         "last": {key: finite_or_text(value) for key, value in last.items()
                                  if key.startswith(("val/", "test/")) and is_number(value) and "/total" not in key}})
        return {"rows": rows}

    def predictions(self, relative, sample=2000, name="predictions.parquet", bins=40, where=None):
        path = self.resolve(relative)
        if path is None or not name.startswith("predictions") or not name.endswith(".parquet"):
            return None
        target = path / name
        if not target.is_file():
            return None
        table = self.frames.read(target)
        total = len(table)
        table, failed = filtered(table, where)
        pairs = prediction_pairs(table)
        named = [{"pred": pred, "target": truth} for pred, truth in pairs]
        targets = [column for column in table.columns if not column.startswith(("pred_", "raw_")) and column != "row"]
        for column in table.columns:
            truth = true_column(column, targets) if column.startswith("pred_") else None
            if truth is not None and column not in {item["pred"] for item in named}:
                named.append({"pred": column, "target": truth})
        found = []
        for pred, truth in pairs:
            actual = table[truth].to_numpy(dtype="float64")
            guess = table[pred].to_numpy(dtype="float64")
            mask = numpy.isfinite(actual) & numpy.isfinite(guess)
            error = guess[mask] - actual[mask]
            counts, edges = numpy.histogram(error, bins=max(2, int(bins))) if len(error) else ([], [])
            shared = shared_histograms(actual[mask], guess[mask], bins) if len(error) else ([], [], [])
            order = numpy.argsort(-numpy.abs(error))[:15]
            rows = table.loc[table.index[mask][order]]
            found.append({"pred": pred, "target": truth, "points": int(mask.sum()),
                          "r2": r2_of(actual[mask], guess[mask]) if len(error) else None,
                          "rmse": float(numpy.sqrt(numpy.mean(error ** 2))) if len(error) else None,
                          "mae": float(numpy.mean(numpy.abs(error))) if len(error) else None,
                          "histogram": {"edges": list(edges), "counts": list(counts)},
                          "distribution": {"edges": list(shared[0]), "data": list(shared[1]),
                                           "pred": list(shared[2])},
                          "worst": [{"row": row["row"], "target": row[truth], "pred": row[pred]}
                                    for _, row in rows[["row", truth, pred]].iterrows()]})
        flags = [column for column in table.columns if column.startswith("flag_")]
        named = {column for pair in pairs for column in pair}
        carried = [column for column in table.columns
                   if column not in named and column not in flags and column != "row"
                   and not column.startswith(("pred_", "raw_"))]
        keep = list(dict.fromkeys(["row", *named, *flags, *carried]))
        whole = sample == "all" or len(table) <= int(sample)
        picked = table if whole else table.sample(int(sample), random_state=0).sort_values("row")
        return clean({"file": name, "rows": len(table), "total": total, "where": where, "error": failed,
                      "carried": carried, "columns": list(table.columns), "pairs": found, "named": named,
                      "flags": flags, "sample": picked[keep].to_dict("records")})

    def classify(self, relative, pred, name="predictions.parquet", score=None, positive=None, bins=40, where=None):
        path = self.resolve(relative)
        if path is None or not pred or not name.startswith("predictions") or not name.endswith(".parquet"):
            return None
        target = path / name
        if not target.is_file():
            return None
        table, failed = filtered(self.frames.read(target), where)
        targets = [column for column in table.columns if not column.startswith(("pred_", "raw_")) and column != "row"]
        truth = true_column(pred, targets)
        if pred not in table.columns or truth is None:
            return None
        found = {"pred": pred, "target": truth, "rows": len(table), "where": where, "error": failed}
        labels = table[truth]
        present = labels.dropna().unique().tolist()
        try:
            classes = sorted(present)
        except TypeError:
            classes = sorted(present, key=str)
        if len(classes) > 100:
            return clean({**found, "error": f"{truth} holds {len(classes)} distinct values; the classification view "
                                            f"takes at most 100 classes"})
        texts = labels.astype(str).to_numpy()
        guess = table[pred]
        known = {str(item) for item in classes}
        decoded = len(guess) > 0 and set(guess.dropna().astype(str).unique().tolist()) <= known
        if decoded:
            hits = guess.astype(str).to_numpy() == texts
            found["accuracy"] = float(hits.mean()) if len(hits) else None
            if len(classes) <= 20:
                guessed = guess.astype(str).to_numpy()
                found["confusion"] = [[int(((texts == str(row)) & (guessed == str(column))).sum())
                                       for column in classes] for row in classes]
        rest = pred[len("pred_"):]
        wire = rest[:-len(truth) - 1] if rest.endswith(f"_{truth}") else rest
        raws = [column for column in table.columns if column.startswith("raw_")]
        own = [column for column in raws if column in (f"raw_{rest}", f"raw_{wire}")
               or re.fullmatch(rf"raw_({re.escape(rest)}|{re.escape(wire)})_\d+", column)]
        choices = [column for column in [*own, *[item for item in raws if item not in own],
                                          *[item for item in table.columns if item.startswith("pred_")]]
                   if table[column].dtype.kind in "fiu"]
        chosen = score if score in choices else None
        picked = next((item for item in classes if positive is not None and str(item) == str(positive)), None)
        found.update({"classes": classes, "counts": [int((texts == str(item)).sum()) for item in classes],
                      "scores": choices, "score": chosen, "positive": picked})
        if chosen is None or picked is None or len(classes) < 2:
            return clean(found)
        values = table[chosen].to_numpy(dtype="float64")
        keep = numpy.isfinite(values) & labels.notna().to_numpy()
        values, signal = values[keep], texts[keep] == str(picked)
        found["roc"] = roc_of(values, signal)
        found["histogram"] = class_histograms(values, texts[keep], classes[:10], bins)
        return clean(found)

    def files(self, relative):
        path = self.resolve(relative)
        if path is None or not path.is_dir():
            return None
        found = [file_note(path, item) for item in sorted(path.rglob("*"))
                 if item.is_file() and "__pycache__" not in item.parts]
        return {"files": found, "total": sum(item["size"] for item in found)}

    def text(self, relative, name, limit=400000):
        path = self.resolve(relative)
        target = (path / name).resolve() if path is not None else None
        if target is None or path not in target.parents or not target.is_file() or target.suffix not in TEXT_SUFFIXES:
            return None
        raw = target.read_bytes()
        return {"name": name, "text": raw[:limit].decode("utf-8", errors="replace"), "truncated": len(raw) > limit,
                "size": len(raw)}

    def prep(self, relative):
        path = self.resolve(relative)
        if path is None or not (path / "fitted" / "preprocessors" / "plan.json").is_file():
            return None
        prep = read_prep(path)
        preprocessors = {}
        for name, entry in prep.fitted.items():
            if isinstance(entry, Grouped):
                preprocessors[name] = {"class": type(entry.preprocessor).__name__, "grouped": True,
                                       "columns": list(entry.columns), "state": small_state(entry.preprocessor)}
            else:
                preprocessors[name] = {"class": next((type(item).__name__ for item in entry.values()), ""),
                                       "grouped": False,
                                       "columns": {column: small_state(item) for column, item in entry.items()}}
        fields = [{"name": item.name, "target": item.target, "chain": list(item.chain), "columns": list(item.columns),
                   "extras": list(item.extras), "dtype": prep.dtypes.get(item.name)} for item in prep.fields]
        return clean({"fields": fields, "preprocessors": preprocessors, "sets": prep.sets, "drop": prep.drop})

    def events(self, relative, limit=20000):
        path = self.resolve(relative)
        if path is None:
            return None
        lines = self.jsonl.read(path / "events.jsonl")
        return {"events": lines[-int(limit):], "total": len(lines)}

    def tree_stamp(self):
        stamps = [stamp(self.root)]
        for child in self.root.iterdir():
            if child.is_dir():
                stamps.append(stamp(child))
                stamps.extend(stamp(grandchild) for grandchild in child.iterdir() if grandchild.is_dir())
        return stamps

    def record_stamp(self, path, files=("manifest.json", "history.jsonl", "steps.jsonl", "events.jsonl", "run.json",
                                        "stdout.txt", "stderr.txt", "resolved.yaml", "data.json", "architecture.json",
                                        "sweep.json", "stop.json", "fitted/calibrate/calibrate.json")):
        snapshot = {name: stamp(path / name) for name in files}
        for folder in ("plots", "samples", "checkpoints", "final", "export"):
            snapshot[folder] = stamp_dir(path / folder)
        for item in path.glob("predictions*.parquet"):
            snapshot[item.name] = stamp(item)
        return snapshot

    def common_stamps(self):
        snapshot = {"tree": self.tree_stamp()}
        for entry in self.records(full=False):
            if entry["kind"] == "sweep" or entry["status"]["state"] not in ("running", "pending"):
                continue
            target = self.root / entry["path"]
            for name in ("history.jsonl", "steps.jsonl", "run.json"):
                snapshot[f"{entry['path']}/{name}"] = stamp(target / name)
        return snapshot

    def watched(self, relative, common=None):
        common = self.common_stamps() if common is None else common
        path = self.resolve(relative) if relative else None
        own = ({f"{path.relative_to(self.root)}/{name}" for name in ("history.jsonl", "steps.jsonl", "run.json")}
               if path is not None else set())
        snapshot = {key: value for key, value in common.items() if key not in own}
        if path is None or not Record(path).is_record:
            return snapshot
        snapshot.update(self.record_stamp(path))
        manifest = self.cache.json(path / "manifest.json") or {}
        if manifest.get("kind") == "sweep":
            for child in sorted(item for item in path.iterdir() if item.is_dir()):
                for name in ("manifest.json", "history.jsonl", "sweep.json", "run.json"):
                    snapshot[f"{child.name}/{name}"] = stamp(child / name)
        return snapshot

    def stop(self, relative):
        path = self.resolve(relative)
        if path is None or not Record(path).is_record:
            return None
        return {"stopped": [relative_to(self.root, Path(item)) for item in api.stop(path, by="board")]}

    def lines(self, relative, name, offset=0):
        path = self.resolve(relative)
        if path is None:
            return None
        lines = self.jsonl.read(path / name)
        return {"lines": [marked(line) for line in lines[int(offset):]], "offset": len(lines)}

    def tail(self, relative, name, count=200):
        path = self.resolve(relative)
        if path is None or name not in ("stdout.txt", "stderr.txt"):
            return None
        target = path / name
        if not target.is_file():
            return {"lines": [], "name": name}
        return {"lines": tail_lines(target, max(1, min(int(count), 20000))), "name": name,
                "total": self.counts.total(target)}

    def best_so_far(self, history, objective):
        monitor = (objective or {}).get("monitor")
        if not monitor or not len(history):
            return None
        try:
            value, turn = history.best(monitor, objective.get("mode", "min"), objective.get("at", "best"))
        except ValueError:
            return None
        return {"value": value, "turn": turn}

    def point(self, relative, objective, full=False, status=None):
        record = Record(self.root / relative, self.cache.json)
        note = record.read_json("manifest.json") or {}
        done = record.read_json("sweep.json")
        if not objective and done is not None:
            objective = {key: done["objective"].get(key) for key in ("monitor", "mode", "at")}
        history = self.history(record.directory) if full or done is None else None
        status = record.status() if full else (status or {})
        point = {"path": relative, "id": done["id"] if done else note.get("id"),
                 "values": done["point"] if done else note.get("values") or {}, "status": status,
                 "objective": {"value": done["objective"]["value"], "turn": done["objective"]["turn"]}
                 if done else self.best_so_far(history, objective)}
        if full:
            point.update({"turns": len(history), "unit": unit_of(note), "started": note.get("started"),
                          "error": failure_text(record) if status.get("state") == "failed" else None})
        return objective, point

    def standing(self, relative, entries):
        manifest = self.cache.json(self.root / relative / "manifest.json") or {}
        objective = manifest.get("objective") or {}
        points = []
        for entry in entries:
            try:
                objective, point = self.point(entry["path"], objective, status=entry["status"])
            except (OSError, ValueError) as problem:
                point = unreadable_point(entry["path"], problem)
            points.append(point)
        return manifest, objective, best_point(points, objective)

    def sweep(self, relative):
        path = self.resolve(relative)
        if path is None:
            return None
        manifest = self.cache.json(path / "manifest.json") or {}
        objective = manifest.get("objective") or {}
        points = []
        for child in sorted(item for item in path.iterdir() if item.is_dir()):
            record = Record(child, self.cache.json)
            relative_child = str(child.relative_to(self.root))
            try:
                if not record.is_record:
                    continue
                if ((record.read_json("manifest.json") or {}).get("kind") or "point") != "point":
                    continue
                objective, point = self.point(relative_child, objective, full=True)
            except (OSError, ValueError) as problem:
                point = unreadable_point(relative_child, problem)
            points.append(point)
        return {"path": relative, "manifest": manifest, "objective": objective, "points": points,
                "best": best_point(points, objective), "space": read_space(manifest, points),
                "unit": sweep_unit(points)}

    def diff(self, first, second):
        paths = [self.resolve(first), self.resolve(second)]
        if any(path is None or not (path / "resolved.yaml").exists() for path in paths):
            return None
        texts = [(path / "resolved.yaml").read_text().splitlines() for path in paths]
        return {"diff": list(difflib.unified_diff(texts[0], texts[1], first, second, lineterm=""))}

    def describe(self, relative):
        from kalfa.api import check
        from kalfa.describe.render import report
        from kalfa.style import Style

        path = self.resolve(relative)
        if path is None or not (path / "resolved.yaml").exists():
            return None
        prepared = check([str(path)])
        return {"text": report(prepared, Style(False))}

    def file(self, relative):
        path = self.resolve(relative)
        if path is None or not path.is_file():
            return None
        return path


def handler_for(board):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *arguments):
            logger.debug(format % arguments)

        def send(self, status, body, kind="application/json"):
            payload = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def stream(self, relative):
            board.watcher.join()
            try:
                self.follow(relative)
            finally:
                board.watcher.leave()

        def follow(self, relative):
            last = board.watched(relative, board.watcher.current())
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            pinged = time.monotonic()
            try:
                self.wfile.write(b": watching\n\n")
                self.wfile.flush()
                while True:
                    time.sleep(board.watcher.every)
                    snapshot = board.watched(relative, board.watcher.current())
                    changed = [name for name in snapshot if snapshot.get(name) != last.get(name)]
                    if changed:
                        self.wfile.write(f"data: {json.dumps({'changed': changed})}\n\n".encode("utf-8"))
                        self.wfile.flush()
                        last = snapshot
                        pinged = time.monotonic()
                    elif time.monotonic() - pinged > 15:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                        pinged = time.monotonic()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return
            except Exception as problem:
                logger.warning(f"watch {relative}: {type(problem).__name__}: {problem}")
                return

        def send_json(self, value):
            if value is None:
                self.send(404, json.dumps({"error": "not found"}))
                return
            self.send(200, json.dumps(clean(value), default=str))

        def send_file(self, found):
            if found is None:
                self.send(404, "not found", "text/plain")
                return
            kind = mimetypes.guess_type(found.name)[0] or "application/octet-stream"
            if kind.startswith("text/") or kind in ("application/javascript", "text/javascript"):
                kind += "; charset=utf-8"
            size = found.stat().st_size
            tag = f'"{size:x}-{found.stat().st_mtime_ns:x}"'
            if self.headers.get("If-None-Match") == tag:
                self.send_response(304)
                self.send_header("ETag", tag)
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("ETag", tag)
            self.end_headers()
            left = size
            with found.open("rb") as stream:
                while left > 0:
                    chunk = stream.read(min(left, 1 << 20))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)

        def do_GET(self):
            url = urlparse(self.path)
            query = {key: values[0] for key, values in parse_qs(url.query).items()}
            try:
                self.route(url, query, query.get("path", ""))
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as problem:
                logger.warning(f"{self.path}: {type(problem).__name__}: {problem}")
                try:
                    self.send(500, json.dumps({"error": f"{type(problem).__name__}: {problem}"}))
                except OSError:
                    return

        def route(self, url, query, path):
            if url.path == "/":
                self.send_file(static_path("index.html"))
            elif url.path.startswith("/static/"):
                self.send_file(static_path(url.path[len("/static/"):]))
            elif url.path == "/api/tree":
                self.send_json(board.tree())
            elif url.path == "/api/live":
                self.send_json(board.live())
            elif url.path == "/api/watch":
                self.stream(path)
            elif url.path == "/api/table":
                self.send_json(board.table())
            elif url.path == "/api/predictions":
                self.send_json(board.predictions(path, query.get("sample", 2000),
                                                 query.get("name", "predictions.parquet"), query.get("bins", 40),
                                                 query.get("where")))
            elif url.path == "/api/series":
                keys = query.get("keys")
                self.send_json(board.series([item for item in query.get("paths", "").split(",") if item],
                                            None if keys is None else [item for item in keys.split(",") if item]))
            elif url.path == "/api/classify":
                self.send_json(board.classify(path, query.get("pred", ""), query.get("name", "predictions.parquet"),
                                              query.get("score"), query.get("positive"), query.get("bins", 40),
                                              query.get("where")))
            elif url.path == "/api/files":
                self.send_json(board.files(path))
            elif url.path == "/api/text":
                self.send_json(board.text(path, query.get("name", "")))
            elif url.path == "/api/prep":
                self.send_json(board.prep(path))
            elif url.path == "/api/events":
                self.send_json(board.events(path))
            elif url.path == "/api/record":
                self.send_json(board.record(path))
            elif url.path == "/api/history":
                self.send_json(board.lines(path, "history.jsonl", query.get("offset", 0)))
            elif url.path == "/api/steps":
                self.send_json(board.lines(path, "steps.jsonl", query.get("offset", 0)))
            elif url.path == "/api/tail":
                self.send_json(board.tail(path, query.get("name", "stdout.txt"), query.get("lines", 200)))
            elif url.path == "/api/sweep":
                self.send_json(board.sweep(path))
            elif url.path == "/api/diff":
                self.send_json(board.diff(query.get("a", ""), query.get("b", "")))
            elif url.path == "/api/describe":
                self.send_json(board.describe(path))
            elif url.path == "/file":
                self.send_file(board.file(path))
            else:
                self.send(404, "not found", "text/plain")

        def do_POST(self):
            url = urlparse(self.path)
            query = {key: values[0] for key, values in parse_qs(url.query).items()}
            if url.path != "/api/stop":
                self.send(404, "not found", "text/plain")
                return
            try:
                self.send_json(board.stop(query.get("path", "")))
            except KalfaError as error:
                self.send(409, json.dumps({"error": str(error)}))

    return Handler


def serve(root, host="127.0.0.1", port=8080):
    server = ThreadingHTTPServer((host, int(port)), handler_for(Board(root)))
    server.daemon_threads = True
    return server
