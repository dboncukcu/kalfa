import contextlib
import json
import os
from pathlib import Path


@contextlib.contextmanager
def atomic(path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{os.getpid()}.tmp"
    try:
        yield temporary
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_text(path, text):
    with atomic(path) as temporary:
        temporary.write_text(text, encoding="utf-8")
    return Path(path)


def write_json(path, mapping):
    return write_text(path, json.dumps(mapping, indent=2, default=str))


def read_json(path):
    target = Path(path)
    if not target.exists():
        return None
    return json.loads(target.read_text(encoding="utf-8"))


def append_line(path, mapping):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(mapping, default=float) + "\n")


def read_lines(path):
    target = Path(path)
    if not target.exists():
        return []
    lines = []
    for text in target.read_text(encoding="utf-8").splitlines():
        if not text.strip():
            continue
        try:
            lines.append(json.loads(text))
        except json.JSONDecodeError:
            break
    return lines
