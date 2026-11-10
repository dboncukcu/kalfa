import ast
import json
import math
import numbers
import os
import sys
from collections.abc import Mapping
from datetime import date, datetime, time
from pathlib import Path, PurePath

import numpy

from kalfa.commands.report import write_report
from kalfa.commands.stop import stop_record
from kalfa.core.errors import KalfaError
from kalfa.records import compare as comparing
from kalfa.records import reader, stats
from kalfa.records import sweep as sweeping


TEXT_SUFFIXES = (".yaml", ".yml", ".json", ".jsonl", ".txt", ".md", ".csv", ".py", ".sh", ".sub", ".log")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".svg")
TEXT_LIMIT = 200_000
EVENTS_LIMIT = 50
TEXT_TYPE = "text/plain; charset=utf-8"
JSON_TYPE = "application/json"
BINARY_TYPE = "application/octet-stream"
IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
               ".svg": "image/svg+xml"}
GALLERIES = ("plots/", "samples/")
STAMPED = (reader.HISTORY, reader.EVENTS)
STATES = ("done", "running", "failed", "stopped", "stale")
SAMPLE = 2000
BINS = 40
MAX_BINS = stats.MAX_BINS
SERIES_LIMIT = 2000
BLOCK = 65_536
MISSING = "-"
WHERE_NODES = (ast.Expression, ast.BoolOp, ast.UnaryOp, ast.BinOp, ast.Compare, ast.Name, ast.Constant, ast.List,
               ast.Tuple, ast.Load, ast.And, ast.Or, ast.Not, ast.Invert, ast.UAdd, ast.USub, ast.Add, ast.Sub,
               ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.BitAnd, ast.BitOr, ast.BitXor, ast.Eq,
               ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot)


class EscapeError(KalfaError):
    pass


class BadValue(KalfaError):
    pass


def segments(relative):
    text = "" if relative is None else str(relative)
    if text.startswith("/") or "\\" in text or "\x00" in text:
        raise EscapeError(f"{text!r} is not a relative posix path")
    parts = text.split("/")
    if parts[-1] == "":
        parts.pop()
    found = []
    for part in parts:
        if part == "..":
            raise EscapeError(f"{text!r} climbs out of the board root")
        if part == "":
            raise EscapeError(f"{text!r} has an empty segment")
        if part != ".":
            found.append(part)
    return found


def inside(base, relative):
    target = base.joinpath(*segments(relative)).resolve()
    if target != base and base not in target.parents:
        raise EscapeError(f"{relative} lies outside {base}")
    return target


def joined(base, name):
    return name if base in ("", ".") else f"{base}/{name}"


def count_of(value, default, label, minimum=None):
    if value is None or value == "":
        return default
    try:
        found = int(value)
    except (TypeError, ValueError) as error:
        raise BadValue(f"{label} must be an integer, not {value!r}") from error
    if minimum is not None and found < minimum:
        raise BadValue(f"{label} must be at least {minimum}, not {value!r}")
    return found


def checked_where(where):
    if where is None or where == "":
        return None
    text = str(where)
    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, ValueError) as error:
        raise BadValue(f"bad where {text!r}: {error}") from error
    for node in ast.walk(tree):
        if not isinstance(node, WHERE_NODES):
            raise BadValue(f"bad where {text!r}: {type(node).__name__} is not allowed")
    return text


def listed(values):
    return [values] if isinstance(values, str) else list(values or [])


def finite_number(value, label):
    try:
        found = float(value)
    except (TypeError, ValueError):
        raise BadValue(f"{label} must be a number, not {value!r}") from None
    if not math.isfinite(found):
        raise BadValue(f"{label} must be a finite number, not {value!r}")
    return found


def histograms_of(histograms):
    if histograms is None or histograms == "" or histograms == {}:
        return None
    if not isinstance(histograms, Mapping):
        raise BadValue("histograms must map histogram names to bin counts, like {\"residual_hist\": 60}")
    return {str(name): count for name, count in histograms.items()}


def overrides_of(pair, **choice):
    found = {key: value for key, value in choice.items() if value is not None and value != ""}
    if "threshold" in found:
        found["threshold"] = finite_number(found["threshold"], "threshold")
    if not found:
        return None
    if pair is None or pair == "":
        named = ["histograms" if key == "bins" else key for key in found]
        raise BadValue(f"{', '.join(named)} needs pair, the prediction column it applies to")
    return {str(pair): found}


def manifest_in(directory):
    return reader.safe_json(directory, reader.MANIFEST)


def image_bases(directory):
    manifest = manifest_in(directory)
    host = manifest.get("host")
    files = manifest.get("config_files")
    found = [directory]
    if isinstance(host, dict) and isinstance(host.get("cwd"), str) and host["cwd"]:
        found.append(Path(host["cwd"]))
    if isinstance(files, list):
        found.extend(Path(file).parent for file in files if isinstance(file, str) and file)
    return [base for base in found if base.is_absolute()]


def image_file(directory, value):
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    for base in image_bases(directory):
        if (base / path).is_file():
            return base / path
    return directory / path


def summary_or_note(directory):
    try:
        return reader.summary_of(directory)
    except (OSError, ValueError) as error:
        return {"path": str(directory), "name": Path(directory).name, "status": "unknown", "error": str(error)}


def counts_of(root):
    states = [reader.state_of(manifest_in(point)) for point in sweeping.point_dirs(root)]
    return {"count": len(states), **{state: states.count(state) for state in STATES}}


def gallery(files):
    return [entry["path"] for entry in files
            if entry["path"].startswith(GALLERIES) and entry["path"].lower().endswith(IMAGE_SUFFIXES)]


def last_lines(path, count):
    if count <= 0:
        return b""
    with path.open("rb") as handle:
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


def clipped(data, size):
    cut = data.find(b"\n")
    if 0 <= cut < len(data) - 1:
        data = data[cut + 1:]
    return f"... {size - len(data)} bytes skipped\n".encode("utf-8") + data


def read_text(path, tail=None):
    if tail is None:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > TEXT_LIMIT:
                handle.seek(size - TEXT_LIMIT)
                data = handle.read(TEXT_LIMIT)
            else:
                data = handle.read()
    else:
        data = last_lines(path, tail)
        size = len(data)
        data = data[-TEXT_LIMIT:]
    if len(data) >= size:
        return data, tail is None
    return clipped(data, size), False


def file_stamp(path):
    try:
        info = path.stat()
    except OSError:
        return MISSING
    return f"{info.st_mtime_ns}:{info.st_size}"


def record_stamp(directory, manifest):
    files = "|".join(file_stamp(directory / name) for name in STAMPED)
    return f"{manifest.get('updated')}|{reader.state_of(manifest)}|{files}"


class Board:
    def __init__(self, root):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise KalfaError(f"{root} is not a directory")

    def resolve(self, relative):
        directory = inside(self.root, relative)
        if not directory.is_dir():
            raise KalfaError(f"no record directory {relative} under {self.root}")
        return directory

    def name_of(self, path):
        return Path(path).relative_to(self.root).as_posix()

    def record_dir(self, relative):
        directory = self.resolve(relative)
        if not reader.is_record(directory):
            raise KalfaError(f"{relative} is not a record directory")
        return directory

    def records(self):
        found = []
        for entry in reader.records_under(self.root):
            directory = self.root / entry["path"]
            if entry["kind"] == "sweep":
                found.append({"path": entry["path"], "kind": "sweep",
                              "sweep": {**entry["manifest"], **counts_of(directory)}})
            else:
                found.append({"path": entry["path"], "kind": "record", "summary": summary_or_note(directory)})
        return {"root": str(self.root), "records": found}

    def record(self, relative):
        directory = self.record_dir(relative)
        manifest = reader.read_manifest(directory)
        plan = reader.read_plan(directory)
        history = reader.read_history(directory)
        files = reader.files_of(directory)
        return {"path": self.name_of(directory), "summary": reader.summary_of(directory), "plan": plan,
                "progress": reader.progress_from(manifest, history), "best": reader.best_from(plan, history),
                "model_summary": reader.model_summary_of(directory), "data_report": reader.data_report_of(directory),
                "events": reader.events_of(directory, limit=EVENTS_LIMIT), "files": files,
                "config": comparing.config_of(directory), "fields": reader.fields_of(directory),
                "has_predictions": (directory / stats.PREDICTIONS).is_file(), "plots": gallery(files),
                "tables": stats.prediction_tables(directory)}

    def series(self, relative, keys=None, limit=SERIES_LIMIT):
        return reader.series_of(self.resolve(relative), keys or None, count_of(limit, SERIES_LIMIT, "limit"))

    def updates(self, relative, keys=None, limit=SERIES_LIMIT):
        return reader.updates_of(self.resolve(relative), keys or None, count_of(limit, SERIES_LIMIT, "limit"))

    def events(self, relative, kinds=None, limit=None):
        return reader.events_of(self.resolve(relative), kinds or None, count_of(limit, None, "limit"))

    def files(self, relative):
        return reader.files_of(self.resolve(relative))

    def file(self, relative, name, tail=None):
        directory = self.resolve(relative)
        path = inside(directory, name)
        if not path.is_file():
            raise KalfaError(f"no file {name} in {self.name_of(directory)}")
        suffix = path.suffix.lower()
        if suffix not in TEXT_SUFFIXES:
            return path.read_bytes(), IMAGE_TYPES.get(suffix, BINARY_TYPE)
        data, whole = read_text(path, count_of(tail, None, "tail"))
        return data, JSON_TYPE if suffix == ".json" and whole else TEXT_TYPE

    def predictions(self, relative, where=None, sample=SAMPLE, bins=BINS, name=stats.PREDICTIONS, pair=None,
                    task=None, score=None, positive=None, threshold=None, histograms=None):
        directory = self.resolve(relative)
        path = inside(directory, name or stats.PREDICTIONS)
        if not path.is_file():
            return None
        options = {"where": checked_where(where), "sample": count_of(sample, SAMPLE, "sample", 0),
                   "bins": min(count_of(bins, BINS, "bins", 1), MAX_BINS),
                   "overrides": overrides_of(pair, task=task, score=score, positive=positive, threshold=threshold,
                                             bins=histograms_of(histograms))}
        try:
            return stats.prediction_stats(directory, plan=reader.read_plan(directory),
                                          name=path.relative_to(directory).as_posix(), **options)
        except KalfaError as error:
            raise BadValue(str(error)) from error

    def image(self, relative, index, name=stats.PREDICTIONS):
        directory = self.resolve(relative)
        path = inside(directory, name or stats.PREDICTIONS)
        table = path.relative_to(directory).as_posix()
        if not path.is_file():
            raise KalfaError(f"no table {table} in {self.name_of(directory)}")
        position = count_of(index, None, "index", 0)
        if position is None:
            raise BadValue("index is required")
        frame = stats.load_table(directory, table)
        if frame is None:
            raise KalfaError(f"{table} cannot be read")
        if position >= len(frame):
            raise BadValue(f"index {position} is past the {len(frame)} rows of {table}")
        value = frame["path"].iloc[position] if "path" in frame.columns else None
        suffix = PurePath(value).suffix.lower() if isinstance(value, str) else ""
        if suffix not in stats.IMAGE_TYPES:
            raise BadValue(f"row {position} of {table} names no image file in its path column")
        found = image_file(directory, value)
        if not found.is_file():
            raise KalfaError(f"no image file {value}")
        return found.read_bytes(), stats.IMAGE_TYPES[suffix]

    def compare(self, relatives, keys=None):
        relatives = listed(relatives)
        if not relatives:
            raise KalfaError("compare needs at least one record")
        directories = [self.record_dir(relative) for relative in relatives]
        found = comparing.compare(directories, keys or None)
        return {**found, "names": [self.name_of(directory) for directory in directories]}

    def sweep(self, relative):
        root = self.resolve(relative)
        if not sweeping.is_sweep(root):
            raise KalfaError(f"{relative} is not a sweep root")
        spec = sweeping.read_sweep(root)
        objective = spec.get("objective") if isinstance(spec, dict) else None
        points = sorted(sweeping.sweep_points(root), key=sweeping.id_key)
        names = []
        for point in points:
            names.extend(key for key in point["values"] if key not in names)
        return {"root": self.name_of(root), "sweep": spec, "points": points,
                "ranked": [point["id"] for point in sweeping.rank_points(points, objective)],
                "best": sweeping.best_point(points, objective),
                "params": {key: [point["values"].get(key) for point in points] for key in names},
                "objectives": [point["objective"] for point in points], "ids": [point["id"] for point in points]}

    def stop(self, relative):
        written = stop_record(self.resolve(relative), by="board")
        return {"paths": [self.name_of(path) for path in written]}

    def report(self, relatives, out=None):
        directories = [self.resolve(relative) for relative in listed(relatives)]
        target = None if out in (None, "") else inside(self.root, out)
        return str(write_report(directories, out=target))

    def stamps(self):
        found = {}
        for entry in reader.records_under(self.root):
            directory = self.root / entry["path"]
            if entry["kind"] != "sweep":
                found[entry["path"]] = record_stamp(directory, entry["manifest"])
                continue
            parts = [file_stamp(directory / reader.SWEEP)]
            for point in sweeping.point_dirs(directory):
                stamp = record_stamp(point, manifest_in(point))
                found[joined(entry["path"], point.name)] = stamp
                parts.append(f"{point.name}={stamp}")
            found[entry["path"]] = ";".join(parts)
        return found


def key_of(key):
    if isinstance(key, str):
        return str(key)
    found = jsonable(key)
    return found if isinstance(found, str) else json.dumps(found)


def ordered(values):
    try:
        return sorted(values)
    except TypeError:
        return list(values)


def is_tensor(value):
    torch = sys.modules.get("torch")
    return torch is not None and isinstance(value, torch.Tensor)


def jsonable(value):
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return str(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {key_of(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [jsonable(item) for item in ordered(value)]
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, numpy.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, numpy.generic):
        return jsonable(value.item())
    if is_tensor(value):
        return jsonable(value.detach().cpu().tolist())
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, numbers.Real):
        return jsonable(float(value))
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return jsonable(tolist())
    return str(value)
