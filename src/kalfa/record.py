import json
import os
import socket
import threading
import time
import traceback
from datetime import datetime
from io import StringIO
from pathlib import Path

from cirak.api import annotated
from ruamel.yaml import YAML

from . import __version__
from .config import written_config
from .std.common.files import append_line, read_json, read_note, write_json, write_text


DATETIME_TOKEN = "$datetime$"
HEARTBEAT_EVERY = 60
LOST_AFTER = 600
ACTIVITY = ("manifest.json", "resolved.yaml", "history.jsonl", "steps.jsonl", "events.jsonl", "stdout.txt",
            "stderr.txt", "heartbeat", "run.json", "failure.json")


class Record:
    def __init__(self, directory, reader=read_json):
        self.directory = Path(directory)
        self.reader = reader

    def path(self, name):
        return self.directory / name

    def write_text(self, name, text):
        return write_text(self.path(name), text)

    def write_json(self, name, mapping):
        return write_json(self.path(name), mapping)

    def read_json(self, name):
        return self.reader(self.path(name))

    def append(self, name, mapping):
        append_line(self.path(name), mapping)

    def manifest(self, kind, **fields):
        existing = self.read_json("manifest.json")
        if existing is not None:
            return existing
        note = {"kind": kind, "started": datetime.now().isoformat(timespec="seconds"), "version": __version__,
                **fields}
        self.write_json("manifest.json", note)
        return note

    def host(self):
        note = {"hostname": socket.gethostname(), "pid": os.getpid(), "cwd": str(Path.cwd())}
        self.write_json("host.json", note)
        return note

    @property
    def is_record(self):
        return self.path("manifest.json").exists() or self.path("resolved.yaml").exists()

    def last_seen(self):
        stamps = []
        for name in ACTIVITY:
            try:
                stamps.append(self.path(name).stat().st_mtime)
            except OSError:
                continue
        return datetime.fromtimestamp(max(stamps)).isoformat(timespec="seconds") if stamps else None

    def state(self):
        ended = self.read_json("run.json")
        if ended is not None:
            return "failed" if (ended.get("status") or "finished") in ("failed", "error") else "finished"
        if self.path("failure.json").exists():
            return "failed"
        if self.lost():
            return "lost"
        if self.path("history.jsonl").exists() or self.path("steps.jsonl").exists():
            return "running"
        return "pending"

    def lost(self):
        try:
            beat = self.path("heartbeat").stat().st_mtime
        except OSError:
            return False
        return time.time() - beat > LOST_AFTER

    def status(self):
        if not self.is_record:
            return None
        return {"state": self.state(), "last_seen": self.last_seen(), "stop": self.stop_note()}

    def request_stop(self, by):
        self.write_json("stop.json", {"by": by, "at": datetime.now().isoformat(timespec="seconds")})

    def stop_requested(self):
        return self.path("stop.json").exists()

    def stop_note(self):
        return read_note(self.path("stop.json"))


def stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def record_dir(config, when=None):
    record = config.get("record")
    if not isinstance(record, str):
        raise ValueError("record must be a path")
    return record.replace(DATETIME_TOKEN, when or stamp())


def resolved_text(surface) -> str:
    yaml = YAML()
    yaml.width = 120
    stream = StringIO()
    yaml.dump(annotated(written_config(surface.data), surface.overrides), stream)
    return stream.getvalue()


def write_resolved(directory, surface):
    Record(directory).write_text("resolved.yaml", resolved_text(surface))


def write_flow(directory, text):
    Record(directory).write_text("flow.yaml", text)


def read_resolved(directory):
    return YAML(typ="safe").load((Path(directory) / "resolved.yaml").read_text())


def resolved_data(surface):
    return YAML(typ="safe").load(resolved_text(surface))


def resume_source(directory):
    last = Path(directory) / "checkpoints" / "last.pt"
    if last.exists():
        return last
    final = Path(directory) / "final" / "state.pt"
    if final.exists():
        return final
    return None


def failure_entry(path, exception):
    return {"path": path, "node": path.rsplit(".", 1)[-1] if path else None, "type": type(exception).__name__,
            "message": str(exception), "traceback": "".join(traceback.format_exception(exception))}


def write_failure(directory, exception):
    failures = getattr(exception, "failures", None)
    if not isinstance(failures, list) or not failures:
        failures = [(None, exception)]
    Record(directory).write_json("failure.json", {"at": datetime.now().isoformat(timespec="seconds"),
                                                  "failures": [failure_entry(path, error)
                                                               for path, error in failures]})


def first_error(node):
    if not isinstance(node, dict):
        return None
    if node.get("error"):
        return node["error"]
    for child in [*(node.get("nodes") or []), *(node.get("turns") or [])]:
        found = first_error(child)
        if found:
            return found
    return None


def failure_text(record):
    try:
        failures = (record.read_json("failure.json") or {}).get("failures") or []
        if failures:
            first = failures[0]
            return f"{first.get('type')}: {first.get('message')}" if first.get("type") else first.get("message")
        return first_error((record.read_json("run.json") or {}).get("tree"))
    except (OSError, ValueError):
        return None


class Heartbeat:
    def __init__(self, directory, every=HEARTBEAT_EVERY):
        self.path = Path(directory) / "heartbeat"
        self.every = every
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.beat, name="kalfa-heartbeat", daemon=True)

    def beat(self):
        while True:
            try:
                self.path.write_text(datetime.now().isoformat(timespec="seconds"))
            except OSError:
                pass
            if self.done.wait(self.every):
                return

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exception):
        self.done.set()
        self.thread.join(timeout=5)
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def write_resume_note(directory, source_run, checkpoint):
    Record(directory).write_json("resume.json", {"resume_from": str(source_run), "checkpoint": str(checkpoint)})


def resume_chain(directory):
    chain = []
    current = Path(directory)
    while (current / "resume.json").exists():
        note = json.loads((current / "resume.json").read_text())
        current = Path(note["resume_from"])
        chain.append(str(current))
        if len(chain) > 100:
            break
    return chain
