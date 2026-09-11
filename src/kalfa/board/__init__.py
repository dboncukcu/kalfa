import difflib
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from kalfa.record import Record
from kalfa.std.common.files import read_lines
from kalfa.std.common.history import History
from kalfa.std.common.log import logger_for


logger = logger_for("board")


def relative_to(root, path):
    return str(Path(path).resolve().relative_to(root))


def static_path(name):
    base = Path(__file__).parent / "static"
    target = (base / (name or "")).resolve()
    if base not in target.parents or not target.is_file():
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
                          "started": manifest.get("started"), "status": record.status()})
        return found

    def tree(self):
        groups = {}
        for entry in self.records():
            parent = str(Path(entry["path"]).parent)
            groups.setdefault(parent, []).append(entry)
        return {"root": str(self.root), "groups": groups}

    def record(self, relative):
        path = self.resolve(relative)
        if path is None or not Record(path).is_record:
            return None
        record = Record(path)
        plots = sorted(item.name for item in (path / "plots").glob("*") if item.is_file())
        samples = sorted(item.name for item in (path / "samples").glob("turn_*.png"))
        resolved = path / "resolved.yaml"
        return {"path": relative, "status": record.status(), "manifest": record.read_json("manifest.json"),
                "host": record.read_json("host.json"), "device": record.read_json("device.json"),
                "git": record.read_json("git.json"), "resume": record.read_json("resume.json"),
                "sweep": record.read_json("sweep.json"), "data": record.read_json("data.json"),
                "run": record.read_json("run.json"), "architecture": record.read_json("architecture.json"),
                "plots": plots, "samples": samples,
                "resolved": resolved.read_text() if resolved.exists() else None,
                "events": read_lines(path / "events.jsonl")[-60:],
                "logs": [name for name in ("stdout.txt", "stderr.txt") if (path / name).is_file()]}

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
        from kalfa.describe import report
        from kalfa.style import Style

        path = self.resolve(relative)
        if path is None or not (path / "resolved.yaml").exists():
            return None
        prepared = check([str(path)])
        return {"text": report(prepared, Style(False))}

    def file(self, relative):
        path = self.resolve(relative)
        if path is None or not path.is_file() or path.parent.name not in ("plots", "samples"):
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

        def send_json(self, value):
            if value is None:
                self.send(404, json.dumps({"error": "not found"}))
                return
            self.send(200, json.dumps(value, default=str))

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

    return Handler


def serve(root, host="127.0.0.1", port=8080):
    server = ThreadingHTTPServer((host, int(port)), handler_for(Board(root)))
    server.daemon_threads = True
    return server
