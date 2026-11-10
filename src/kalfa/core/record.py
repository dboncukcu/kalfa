import importlib.util
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime
from io import StringIO
from pathlib import Path

import torch
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from .errors import RunError
from .refs import tokenize


FILES = ("manifest.json", "resolved.yaml", "plan.json", "history.jsonl", "updates.jsonl", "events.jsonl")
STOP_FILE = "stop.json"


def file_mode():
    mask = os.umask(0)
    os.umask(mask)
    return 0o666 & ~mask


MODE = file_mode()


@contextmanager
def atomic(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(handle)
    try:
        yield Path(temporary)
        os.chmod(temporary, MODE)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def now():
    return datetime.now().isoformat(timespec="seconds")


def version():
    from kalfa import __version__

    return __version__


class RecordWriter:
    def __init__(self, directory):
        self.directory = Path(directory)

    def path(self, name):
        return self.directory / name

    def write_json(self, name, value):
        with atomic(self.path(name)) as temporary:
            temporary.write_text(json.dumps(value, indent=1, default=str), encoding="utf-8")

    def write_text(self, name, text):
        with atomic(self.path(name)) as temporary:
            temporary.write_text(text, encoding="utf-8")

    def read_json(self, name):
        path = self.path(name)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def append(self, name, line):
        path = self.path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, default=str) + "\n")

    def event(self, kind, turn=None, **fields):
        self.append("events.jsonl", {"at": now(), "turn": turn, "kind": kind, **fields})

    def manifest(self, **changes):
        note = self.read_json("manifest.json") or {}
        note.update(changes)
        note["updated"] = now()
        self.write_json("manifest.json", note)
        return note

    def stop_requested(self):
        return self.path(STOP_FILE).exists()

    def stop_note(self):
        return self.read_json(STOP_FILE)


def commented(value):
    if isinstance(value, dict):
        out = CommentedMap()
        for key, item in value.items():
            out[key] = commented(item)
        return out
    if isinstance(value, (list, tuple)):
        out = CommentedSeq()
        for item in value:
            out.append(commented(item))
        return out
    return value


def locate(document, tokens):
    holder = document
    for token in tokens[:-1]:
        try:
            holder = holder[token]
        except (KeyError, IndexError, TypeError):
            return None, None
    if not tokens or not isinstance(holder, (CommentedMap, CommentedSeq)):
        return None, None
    try:
        holder[tokens[-1]]
    except (KeyError, IndexError, TypeError):
        return None, None
    return holder, tokens[-1]


def annotate(document, overrides):
    for path, label in (overrides or {}).items():
        try:
            tokens = tokenize(path)
        except RunError:
            continue
        holder, key = locate(document, tokens)
        if holder is None:
            continue
        try:
            holder.yaml_add_eol_comment(str(label), key=key)
        except (KeyError, IndexError, TypeError, AttributeError):
            try:
                holder.yaml_add_eol_comment(str(label), key=key, column=0)
            except (KeyError, IndexError, TypeError, AttributeError):
                continue
    return document


def resolved_mapping(config, record=None):
    data = {key: value for key, value in (config.data or {}).items() if key != "sweep"}
    if record is not None:
        data["record"] = str(record)
    plugins = [str(name) for name in (getattr(config, "plugins", None) or [])]
    if plugins and "plugins" not in data:
        return {"plugins": plugins, **data}
    return data


def resolved_text(config, record=None):
    yaml = YAML()
    yaml.width = 120
    document = annotate(commented(resolved_mapping(config, record)), getattr(config, "overrides", None))
    stream = StringIO()
    yaml.dump(document, stream)
    return stream.getvalue()


def host_note():
    return {"hostname": socket.gethostname(), "platform": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__, "cuda": torch.version.cuda or None, "cwd": str(Path.cwd())}


def git_note(root):
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root), capture_output=True, text=True,
                                check=True, timeout=5)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=str(root), capture_output=True, text=True,
                                check=True, timeout=5)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return {"commit": commit.stdout.strip(), "dirty": bool(status.stdout.strip())}


def start_record(ctx, directory, paths=(), plan=None, parent=None, sweep=None):
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()) and parent is None:
        raise RunError(f"record directory {directory} exists and is not empty; name another record or resume")
    directory.mkdir(parents=True, exist_ok=True)
    writer = RecordWriter(directory)
    config = ctx.config
    train = getattr(ctx, "train", None)
    device = getattr(ctx, "device", None)
    writer.manifest(format=1, kalfa=version(), name=directory.resolve().name, started=now(), finished=None,
                    status="running", config_files=[str(Path(path).resolve()) for path in paths],
                    params=dict(getattr(config, "params", None) or {}),
                    unit=train.unit if train is not None else "epochs",
                    turns_planned=train.total_turns() if train is not None else 0, host=host_note(),
                    device=str(device) if device is not None else None, git=git_note(Path.cwd()), sweep=sweep,
                    resumed_from=str(parent) if parent is not None else None, failure=None)
    writer.write_text("resolved.yaml", resolved_text(config, record=directory))
    if plan is not None:
        writer.write_json("plan.json", plan)
    copy_plugins(config, directory, imports=getattr(getattr(ctx, "tree", None), "imports", None) or ())
    ctx.record = writer
    return writer


def module_file(name, search=False):
    module = sys.modules.get(name)
    file = getattr(module, "__file__", None) if module is not None else None
    if file is None and search and module is None:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError, AttributeError):
            return None
        file = spec.origin if spec is not None and spec.has_location else None
    if not file or not str(file).endswith(".py"):
        return None
    return Path(file).resolve()


def plugin_source(name, roots, search=False):
    parts = str(name).split(".")
    for cut in range(1, len(parts) + 1):
        file = module_file(".".join(parts[:cut]), search=search)
        if file is None:
            continue
        home = next((root for root in roots if file.is_relative_to(root)), None)
        if home is not None:
            return file, home
    return (module_file(name, search=search) if search else None), None


def copy_module(file, home, destination):
    relative = file.relative_to(home) if home is not None else Path(file.name)
    if file.name == "__init__.py":
        target = destination / (relative.parent if home is not None else Path(file.parent.name))
        shutil.copytree(file.parent, target, ignore=shutil.ignore_patterns("__pycache__"), dirs_exist_ok=True)
        return target
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(file, target)
    return target


def manifest_files(directory):
    try:
        files = json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8")).get("config_files") or []
    except (OSError, ValueError, AttributeError):
        return []
    return [str(file) for file in files if isinstance(file, str)]


def plugin_roots(config):
    roots = []
    for item in getattr(config, "directories", None) or []:
        root = Path(item).resolve()
        if (root / "resolved.yaml").is_file():
            if (root / "plugins").is_dir():
                roots.append(root / "plugins")
            roots.extend(Path(file).resolve().parent for file in manifest_files(root))
        roots.append(root)
    return list(dict.fromkeys(roots))


def copy_plugins(config, directory, imports=()):
    roots = plugin_roots(config)
    named = [str(name) for name in (getattr(config, "plugins", None) or [])]
    destination = Path(directory) / "plugins"
    copied = []
    seen = set()
    for name, search in [(name, True) for name in named] + [(str(text), False) for text in imports or ()]:
        file, home = plugin_source(name, roots, search=search)
        if file is None or file in seen:
            continue
        seen.add(file)
        copied.append(copy_module(file, home, destination))
    return copied
