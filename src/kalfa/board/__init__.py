import difflib
import json
import math
import mimetypes
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy
import pandas

from kalfa import api
from kalfa.errors import KalfaError
from kalfa.record import Record, read_resolved
from kalfa.std.common.files import read_json, read_lines
from kalfa.std.common.history import History
from kalfa.std.common.log import logger_for
from kalfa.std.plot.base import prediction_pairs, r2_of
from kalfa.std.pre.base import Grouped, read_prep


logger = logger_for("board")

TEXT_SUFFIXES = (".yaml", ".yml", ".json", ".jsonl", ".txt", ".md", ".csv", ".tsv", ".py", ".sh", ".sub", ".plan",
                 ".log", ".toml", ".ini", ".cfg", ".rst", ".html", ".xml")


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


def sweep_state(states):
    if "running" in states:
        return "running"
    if states and all(state == "finished" for state in states):
        return "finished"
    if "failed" in states:
        return "failed"
    return "pending"


def settle(entries):
    below = {}
    for entry in entries:
        below.setdefault(str(Path(entry["path"]).parent), []).append(entry["status"]["state"])
    for entry in entries:
        states = below.get(entry["path"]) if entry["kind"] == "sweep" else None
        if states:
            entry["status"] = {**entry["status"], "state": sweep_state(states)}
    return entries


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
    base = (Path(__file__).parent / "static").resolve()
    target = (base / (name or "")).resolve()
    if not target.is_relative_to(base) or not target.is_file():
        return None
    return target


class Board:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def resolve(self, relative):
        target = (self.root / (relative or "")).resolve()
        if target != self.root and self.root not in target.parents:
            return None
        return target

    def records(self):
        found = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_dir():
                continue
            record = Record(path)
            if not record.is_record:
                continue
            manifest = record.read_json("manifest.json") or {}
            kind = manifest.get("kind") or "run"
            if kind == "data":
                continue
            found.append({"path": relative_to(self.root, path), "kind": kind, "name": manifest.get("name") or path.name,
                          "started": manifest.get("started"), "status": record.status(),
                          "unit": "epoch" if manifest.get("turn") == "epoch" else "turn"})
        return settle(found)

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
        record = Record(path)
        plots = sorted(item.name for item in (path / "plots").glob("*") if item.is_file())
        samples = sorted(item.name for item in (path / "samples").glob("turn_*.png"))
        resolved = path / "resolved.yaml"
        config = read_resolved(path) if resolved.exists() else None
        checkpoints = [file_note(path, item) for folder in ("checkpoints", "final", "export")
                       for item in sorted((path / folder).glob("*")) if item.is_file()]
        return {"path": relative, "status": record.status(), "manifest": record.read_json("manifest.json"),
                "best": self.best_of(config, History.read(path)), "checkpoints": checkpoints,
                "calibrations": read_json(path / "fitted" / "calibrate" / "calibrate.json"),
                "predictions": sorted(item.name for item in path.glob("predictions*.parquet")),
                "host": record.read_json("host.json"), "device": record.read_json("device.json"),
                "git": record.read_json("git.json"), "resume": record.read_json("resume.json"),
                "sweep": record.read_json("sweep.json"), "data": record.read_json("data.json"),
                "run": record.read_json("run.json"), "architecture": record.read_json("architecture.json"),
                "plots": plots, "samples": samples,
                "resolved": resolved.read_text() if resolved.exists() else None,
                "config": config,
                "events": read_lines(path / "events.jsonl")[-60:],
                "logs": [name for name in ("stdout.txt", "stderr.txt") if (path / name).is_file()]}

    def progress(self, path):
        config = read_resolved(path) if (path / "resolved.yaml").exists() else {}
        training = (config or {}).get("training") or {}
        history = History.read(path)
        last = history[-1] if len(history) else {}
        seconds = [line["seconds"] for line in history if isinstance(line.get("seconds"), (int, float))]
        planned = turns_planned(training)
        remaining = planned - len(history) if planned is not None else None
        eta = sum(seconds) / len(seconds) * remaining if seconds and remaining is not None and remaining > 0 else None
        step = last_line(path / "steps.jsonl") or {}
        report = read_json(path / "data.json") or {}
        batches = ((report.get("loaders") or {}).get("train") or {}).get("batches")
        monitor = monitor_key(training)
        shown = {key: value for key, value in last.items()
                 if key.startswith(("val/", "test/")) and isinstance(value, (int, float)) and "/total" not in key}
        current = step.get("step")
        return {"turn": len(history), "turns_total": planned, "eta": eta, "monitor": monitor,
                "monitor_value": last.get(monitor) if monitor else None, "step": current,
                "step_in_turn": current - last.get("global_step", 0) if current is not None else None,
                "steps_per_turn": batches,
                "losses": {key: value for key, value in step.items() if key.startswith("loss/")},
                "last": dict(list(shown.items())[:8])}

    def live(self):
        live, recent = [], []
        for entry in self.records():
            path = self.resolve(entry["path"])
            if entry["kind"] == "sweep":
                sweep = self.sweep(entry["path"])
                states = [point["status"]["state"] for point in sweep["points"]]
                active = any(state in ("running", "pending") for state in states)
                total = sweep["manifest"].get("total") or len(states)
                note = {**entry, "finished": states.count("finished"), "running": states.count("running"),
                        "failed": states.count("failed"), "total": total,
                        "objective": sweep["objective"], "best": sweep["best"]}
                (live if active else recent).append(note)
                continue
            if entry["status"]["state"] in ("running", "pending"):
                live.append({**entry, **self.progress(path)})
            else:
                recent.append(entry)
        recent.sort(key=lambda item: item["status"].get("last_seen") or "", reverse=True)
        return {"live": live, "recent": recent[:12]}

    def table(self):
        rows = []
        for entry in self.records():
            if entry["kind"] == "sweep":
                continue
            path = self.resolve(entry["path"])
            record = Record(path)
            config = read_resolved(path) if (path / "resolved.yaml").exists() else None
            history = History.read(path)
            last = history[-1] if len(history) else {}
            manifest = record.read_json("manifest.json") or {}
            rows.append({**entry, "params": manifest.get("params") or {}, "turns": len(history),
                         "seconds": sum(line["seconds"] for line in history if is_number(line.get("seconds"))),
                         "best": self.best_of(config, history),
                         "device": (record.read_json("device.json") or {}).get("device"),
                         "last": {key: value for key, value in last.items()
                                  if key.startswith(("val/", "test/")) and is_number(value) and "/total" not in key}})
        return {"rows": rows}

    def predictions(self, relative, sample=2000, name="predictions.parquet", bins=40, where=None):
        path = self.resolve(relative)
        if path is None or not name.startswith("predictions") or not name.endswith(".parquet"):
            return None
        target = path / name
        if not target.is_file():
            return None
        table = pandas.read_parquet(target)
        total, failed = len(table), None
        if where:
            try:
                table = table.query(where)
            except Exception as problem:
                failed = f"{type(problem).__name__}: {problem}"
                table = table.iloc[:0]
        pairs = prediction_pairs(table)
        found = []
        for pred, truth in pairs:
            actual = table[truth].to_numpy(dtype="float64")
            guess = table[pred].to_numpy(dtype="float64")
            mask = numpy.isfinite(actual) & numpy.isfinite(guess)
            error = guess[mask] - actual[mask]
            counts, edges = numpy.histogram(error, bins=max(2, int(bins))) if len(error) else ([], [])
            order = numpy.argsort(-numpy.abs(error))[:15]
            rows = table.loc[table.index[mask][order]]
            found.append({"pred": pred, "target": truth, "points": int(mask.sum()),
                          "r2": r2_of(actual[mask], guess[mask]) if len(error) else None,
                          "rmse": float(numpy.sqrt(numpy.mean(error ** 2))) if len(error) else None,
                          "mae": float(numpy.mean(numpy.abs(error))) if len(error) else None,
                          "histogram": {"edges": list(edges), "counts": list(counts)},
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
                      "carried": carried, "columns": list(table.columns), "pairs": found, "flags": flags,
                      "sample": picked[keep].to_dict("records")})

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
        lines = read_lines(path / "events.jsonl")
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

    def watched(self, relative):
        snapshot = {"tree": self.tree_stamp()}
        path = self.resolve(relative) if relative else None
        for entry in self.live()["live"]:
            target = self.resolve(entry["path"])
            if entry["kind"] == "sweep" or target == path:
                continue
            for name in ("history.jsonl", "steps.jsonl", "run.json"):
                snapshot[f"{entry['path']}/{name}"] = stamp(target / name)
        if path is None or not Record(path).is_record:
            return snapshot
        snapshot.update(self.record_stamp(path))
        manifest = Record(path).read_json("manifest.json") or {}
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
        lines = read_lines(path / name)
        return {"lines": lines[int(offset):], "offset": len(lines)}

    def tail(self, relative, name, count=200):
        path = self.resolve(relative)
        if path is None or name not in ("stdout.txt", "stderr.txt"):
            return None
        target = path / name
        if not target.is_file():
            return {"lines": [], "name": name}
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        return {"lines": lines[-int(count):], "name": name, "total": len(lines)}

    def best_so_far(self, history, objective):
        monitor = (objective or {}).get("monitor")
        if not monitor or not len(history):
            return None
        try:
            value, turn = history.best(monitor, objective.get("mode", "min"), objective.get("at", "best"))
        except ValueError:
            return None
        return {"value": value, "turn": turn}

    def sweep(self, relative):
        path = self.resolve(relative)
        if path is None:
            return None
        manifest = Record(path).read_json("manifest.json") or {}
        objective = manifest.get("objective") or {}
        points = []
        for child in sorted(item for item in path.iterdir() if item.is_dir()):
            record = Record(child)
            if not record.is_record:
                continue
            note = record.read_json("manifest.json") or {}
            if (note.get("kind") or "point") != "point":
                continue
            done = record.read_json("sweep.json")
            history = History.read(child)
            if not objective and done is not None:
                objective = {key: done["objective"].get(key) for key in ("monitor", "mode", "at")}
            points.append({"path": relative_to(self.root, child), "id": done["id"] if done else note.get("id"),
                           "values": done["point"] if done else note.get("values") or {},
                           "status": record.status(), "turns": len(history),
                           "objective": {"value": done["objective"]["value"], "turn": done["objective"]["turn"]}
                           if done else self.best_so_far(history, objective)})
        scored = [point for point in points if point["objective"] is not None]
        pick = min if objective.get("mode", "min") == "min" else max
        best = pick(scored, key=lambda point: point["objective"]["value"]) if scored else None
        return {"path": relative, "manifest": manifest, "objective": objective, "points": points, "best": best}

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
            last = board.watched(relative)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            pinged = time.monotonic()
            try:
                self.wfile.write(b": watching\n\n")
                self.wfile.flush()
                while True:
                    time.sleep(1.0)
                    snapshot = board.watched(relative)
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
            self.send(200, found.read_bytes(), kind)

        def do_GET(self):
            url = urlparse(self.path)
            query = {key: values[0] for key, values in parse_qs(url.query).items()}
            path = query.get("path", "")
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
