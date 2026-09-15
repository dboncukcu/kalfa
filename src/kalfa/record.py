import json
import os
import socket
from datetime import datetime
from io import StringIO
from pathlib import Path

from cirak.api import annotated
from ruamel.yaml import YAML

from . import __version__
from .config import written_config
from .std.common.files import append_line, read_json, read_note, write_json, write_text


DATETIME_TOKEN = "$datetime$"


class Record:
    def __init__(self, directory):
        self.directory = Path(directory)

    def path(self, name):
        return self.directory / name

    def write_text(self, name, text):
        return write_text(self.path(name), text)

    def write_json(self, name, mapping):
        return write_json(self.path(name), mapping)

    def read_json(self, name):
        return read_json(self.path(name))

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
        stamps = [path.stat().st_mtime for path in self.directory.iterdir() if path.is_file()]
        return datetime.fromtimestamp(max(stamps)).isoformat(timespec="seconds") if stamps else None

    def state(self):
        ended = self.read_json("run.json")
        if ended is not None:
            return "failed" if (ended.get("status") or "finished") in ("failed", "error") else "finished"
        if self.path("history.jsonl").exists() or self.path("steps.jsonl").exists():
            return "running"
        return "pending"

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


def resume_source(directory):
    last = Path(directory) / "checkpoints" / "last.pt"
    if last.exists():
        return last
    final = Path(directory) / "final" / "state.pt"
    if final.exists():
        return final
    return None


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
