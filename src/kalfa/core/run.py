import contextlib
import logging
import os
import shutil
import signal
import socket
import sys
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas
import torch

from ..records.reader import read_lines
from . import build, loop, predict
from .aliases import add_search_dir
from .check import check
from .context import Context
from .device import Device
from .errors import KalfaError, ProblemsError, RunError
from .history import progress_text
from .record import Pulse, RecordWriter, atomic, manifest_files, now, start_record
from .rng import Rng
from .rules import Rule, replay
from .spec import parse_param, parse_set
from .state import checkpoint_path, load_checkpoint


CHECKPOINTS = ("best", "last", "final")
TABLES = (".parquet", ".pq", ".csv", ".tsv")


@dataclass
class Result:
    record: str
    history: list
    context: object


class Monitor:
    def __init__(self, stream=None, quiet=False):
        self.stream = stream
        self.quiet = quiet
        self.total = None

    def expect(self, total):
        self.total = total

    def say(self, text):
        if not self.quiet:
            print(text, file=self.stream or sys.stderr, flush=True)

    def turn(self, line):
        self.say(progress_text(line, self.total))


def stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def default_record(paths):
    files = [str(path) for path in paths if not str(path).startswith("<")]
    stem = Path(files[0]).stem if files else "run"
    return f"runs/{stem}_{stamp()}"


def pairs(items, kind="set"):
    parser = parse_param if kind == "param" else parse_set
    flag = "-p" if kind == "param" else "--set"
    found = []
    if isinstance(items, dict):
        items = [items]
    for item in items or ():
        if isinstance(item, str):
            found.append(tuple(parser(item)))
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            found.append((str(item[0]), item[1]))
        elif isinstance(item, dict):
            found.extend((str(key), value) for key, value in item.items())
        else:
            raise KalfaError(f"{flag} takes 'NAME=VALUE' or a (name, value) pair, got {item!r}")
    return found


def config_files(checked, paths=()):
    given = [str(path) for path in paths]
    layers = getattr(checked.config, "layers", None) or []
    given.extend(str(file) for file in (getattr(layer, "file", None) for layer in layers) if file)
    for file in list(given):
        if Path(file).name == "resolved.yaml":
            given.extend(manifest_files(Path(file).parent))
    files, seen = [], set()
    for file in given:
        key = str(Path(file).resolve())
        if key not in seen:
            seen.add(key)
            files.append(file)
    return files


def note_default(ctx, path, key, value):
    nodes = getattr(ctx.tree, "nodes", None) or {}
    node = nodes.get(path)
    if node is not None and isinstance(node.keys, dict) and node.keys.get(key) is None:
        node.keys[key] = value
    planned = ctx.plan.get("nodes") if isinstance(ctx.plan, dict) else None
    entry = planned.get(path) if isinstance(planned, dict) else None
    if isinstance(entry, dict) and isinstance(entry.get("keys"), dict) and entry["keys"].get(key) is None:
        entry["keys"][key] = value


def fill_defaults(ctx):
    predicts = ctx.predicts()
    targets = predict.target_names(ctx.data)
    for section, entries in (("losses", ctx.losses), ("metrics", ctx.metrics)):
        for name, entry in entries.items():
            if not hasattr(entry, "pred") or not hasattr(entry, "target"):
                continue
            path = f"{section}.{name}"
            if entry.pred is None:
                if predicts is None or not getattr(predicts, "outputs", None):
                    raise RunError(f"{path}: pred_required: there is no predicts model whose first output could be "
                                   f"pred; name pred")
                entry.pred = predicts.outputs[0]
                note_default(ctx, path, "pred", entry.pred)
            if entry.target is None:
                if len(targets) != 1:
                    raise RunError(f"{path}: target_required: the data has {len(targets)} targets, not one; name "
                                   f"target")
                entry.target = targets[0]
                note_default(ctx, path, "target", entry.target)
            if section == "metrics" and getattr(entry, "rescale", None) is None:
                entry.rescale = predict.rescales(ctx.data, entry.target)
                note_default(ctx, path, "rescale", entry.rescale)


def node_object(tree, path):
    node = tree.nodes.get(path)
    return None if node is None else node.obj


def build_rules(ctx, tree, config):
    ctx.rules = []
    ctx.rules_by_name = {}
    for spec in tree.rules:
        rule = Rule(spec, when=node_object(tree, f"rules.{spec.name}.when"),
                    run=node_object(tree, f"rules.{spec.name}.run"))
        ctx.rules.append(rule)
        ctx.rules_by_name[spec.name] = rule


def place(ctx):
    device = ctx.device.torch
    for entry in ctx.models.values():
        module = getattr(entry, "module", entry)
        module.to(device)
        ema = getattr(entry, "ema", None)
        if isinstance(ema, torch.nn.Module):
            ema.to(device)
    for objective in ctx.losses.values():
        if isinstance(objective, torch.nn.Module):
            objective.to(device)
    for metric in ctx.metrics.values():
        mover = getattr(metric, "to", None)
        if callable(mover):
            mover(device)
    amp = bool(getattr(ctx.step, "amp", False))
    for entry in ctx.optimizers.values():
        if amp:
            scaler = ctx.device.scaler(True)
            if scaler is not None:
                entry.scaler = scaler
        if getattr(entry, "real", None) is None:
            entry.build()


def save_fitted(ctx, directory):
    saver = getattr(ctx.data, "save", None)
    if callable(saver) and not (Path(directory) / "fitted").exists():
        saver(str(directory))


def data_document(ctx):
    from kalfa.std.data import flow

    data = ctx.data
    report = None
    reporter = getattr(data, "report", None)
    if callable(reporter):
        try:
            report = reporter()
        except Exception as exc:
            ctx.logger.warning(f"data.json: the data report failed ({type(exc).__name__}: {exc})")
    try:
        pipeline = flow.pipeline_of(data)
    except Exception as exc:
        pipeline = None
        ctx.logger.warning(f"data.json: the data flowchart could not be drawn ({type(exc).__name__}: {exc}); the "
                           f"board shows the data tables without it")
    return flow.jsonable({"format": 1, "kind": flow.kind_of(data), "report": report, "pipeline": pipeline})


def save_data(ctx):
    if ctx.record is not None and getattr(ctx, "data", None) is not None:
        ctx.record.write_json("data.json", data_document(ctx))


def unused_directory(directory):
    directory = Path(directory)
    if not directory.exists() or not any(directory.iterdir()):
        return directory
    base = f"{str(directory).rstrip('/')}_{stamp()}"
    chosen = Path(base)
    count = 1
    while chosen.exists():
        count += 1
        chosen = Path(f"{base}_{count}")
    logging.getLogger("kalfa.run").warning(f"record {directory} already holds a run; writing {chosen}")
    return chosen


def prepare(checked, record=None, monitor=None, parent=None, paths=(), fitted=None, sets=True, sweep=None):
    if checked.errors:
        raise ProblemsError(checked.problems)
    tree = checked.tree
    config = checked.config.data
    files = config_files(checked, paths)
    ctx = Context(tree, checked.plan)
    ctx.monitor = monitor if monitor is not None else Monitor(quiet=True)
    ctx.seed = int(config.get("seed") or 0)
    ctx.rng = Rng(ctx.seed)
    ctx.rng.seed_everything()
    ctx.device = Device(config.get("device") or "auto")
    ctx.fitted_from = fitted
    ctx.want_sets = sets
    directory = None if record is False else Path(record or config.get("record") or default_record(files))
    if directory is not None and parent is None and sweep is None:
        directory = unused_directory(directory)
    build.instantiate(tree, ctx)
    fill_defaults(ctx)
    build_rules(ctx, tree, config)
    place(ctx)
    expect = getattr(ctx.monitor, "expect", None)
    if callable(expect):
        expect(ctx.train.total_turns())
    if directory is not None:
        writer = start_record(ctx, directory, files, checked.plan, parent=parent, sweep=sweep)
        if ctx.record is None:
            ctx.record = writer if isinstance(writer, RecordWriter) else RecordWriter(directory)
        save_fitted(ctx, directory)
        save_data(ctx)
        say = getattr(ctx.monitor, "say", None)
        if callable(say):
            say(f"record {directory}")
    return ctx


@contextlib.contextmanager
def interruptible(ctx):
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    asked = []

    def handle(number, frame):
        if asked:
            raise KeyboardInterrupt
        asked.append(number)
        ctx.stop("keyboard")
        ctx.stopped_by = "keyboard"
        say = getattr(ctx.monitor, "say", None)
        if callable(say):
            say("stop requested: the run ends after this turn with its final state and predictions; ctrl-c again "
                "aborts it now")

    try:
        previous = signal.signal(signal.SIGINT, handle)
    except ValueError:
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


def finish(ctx, status):
    if ctx.record is not None:
        extra = {}
        if status == "stopped" and getattr(ctx, "stopping", None):
            extra = {"stop_reason": ctx.stopping, "stopped_by": getattr(ctx, "stopped_by", None)}
        ctx.record.manifest(status=status, finished=now(), **extra)


def note_record(ctx, exc):
    if ctx.record is not None:
        exc.add_note(f"record {ctx.record.directory}")


class RecordLog(logging.Handler):
    def __init__(self, record):
        super().__init__(level=logging.INFO)
        self.record = record

    def emit(self, entry):
        try:
            self.record.log(f"{entry.levelname.lower()}: {entry.getMessage()}")
        except Exception:
            self.handleError(entry)


@contextlib.contextmanager
def followed(ctx):
    record = ctx.record
    if record is None:
        yield
        return
    logger = logging.getLogger("kalfa")
    handler = RecordLog(record)
    logger.addHandler(handler)
    record.log(f"run on {socket.gethostname()}, pid {os.getpid()}, device {getattr(ctx, 'device', None)}")
    try:
        with Pulse(record):
            yield
    finally:
        logger.removeHandler(handler)


def train_into(ctx):
    with followed(ctx):
        return trained(ctx)


def trained(ctx):
    history = None
    try:
        with interruptible(ctx):
            history = loop.train(ctx)
        loop.after(ctx)
        finish(ctx, "done" if ctx.stopping is None else "stopped")
    except KeyboardInterrupt as exc:
        if ctx.record is not None:
            ctx.record.event("stop", turn=ctx.turn, by="keyboard", aborted=True)
            ctx.record.log(f"aborted from the keyboard in turn {ctx.turn}")
        finish(ctx, "stopped")
        note_record(ctx, exc)
        raise
    except Exception as exc:
        if ctx.record is not None:
            failure = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
            ctx.record.event("error", turn=ctx.turn, type=failure["type"], message=failure["message"])
            ctx.record.manifest(status="failed", finished=now(), failure=failure)
            ctx.record.log(f"failed in turn {ctx.turn}: {failure['type']}: {failure['message']}")
            ctx.record.log(failure["traceback"].rstrip())
        note_record(ctx, exc)
        raise
    if ctx.record is not None:
        ctx.record.log(f"{'done' if ctx.stopping is None else 'stopped: ' + str(ctx.stopping)} after turn {ctx.turn}")
    if history is None:
        history = ctx.history
    record = str(ctx.record.directory) if ctx.record is not None else None
    return Result(record=record, history=list(history), context=ctx)


def run(paths=(), sets=(), params=(), mappings=(), record=None, monitor=None, resume_from=None, sweep=None):
    if resume_from is not None:
        return resume(resume_from, sets=sets, params=params, target=record, monitor=monitor, paths=paths)
    files = tuple(str(path) for path in paths)
    checked = check(paths=files, sets=pairs(sets), params=pairs(params, "param"), mappings=list(mappings))
    ctx = prepare(checked, record, monitor, paths=files, sweep=sweep)
    return train_into(ctx)


def config_directories(source):
    found = []
    for file in manifest_files(source):
        directory = Path(file).parent
        if directory.is_dir() and directory not in found:
            found.append(directory)
    return found


def record_root(record):
    source = Path(record)
    if not (source / "resolved.yaml").exists():
        raise KalfaError(f"{record} has no resolved.yaml; it is no record directory")
    plugins = source / "plugins"
    if plugins.is_dir():
        add_search_dir(plugins)
    for directory in config_directories(source):
        add_search_dir(directory)
    return source


def carry_lines(source, ctx, upto):
    for line in read_lines(source / "history.jsonl"):
        turn = line.get("turn")
        if isinstance(turn, (int, float)) and turn <= upto:
            ctx.history.append(line)
            ctx.record.append("history.jsonl", line)
    for line in read_lines(source / "updates.jsonl"):
        turn = line.get("turn")
        if isinstance(turn, (int, float)) and turn <= upto:
            ctx.record.append("updates.jsonl", line)


def fresh_directory(source, target):
    directory = Path(target) if target else Path(f"{str(source).rstrip('/')}_{stamp()}")
    if directory.exists() and any(directory.iterdir()):
        same = directory.resolve() == source.resolve()
        what = "the record itself" if same else "a directory that is not empty"
        raise RunError(f"a resumed run continues into a new directory, not into {what} ({directory}); name an empty "
                       f"one with --record or leave it out for {source}_<stamp>")
    return directory


def carry_best(source, ctx):
    best = source / "checkpoints" / "best.pt"
    if best.exists():
        target = ctx.record.path("checkpoints/best.pt")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(best, target)


def carry_plugins(source, ctx):
    plugins = source / "plugins"
    if plugins.is_dir():
        shutil.copytree(plugins, ctx.record.path("plugins"), ignore=shutil.ignore_patterns("__pycache__"),
                        dirs_exist_ok=True)


def carry_params(source, ctx):
    before = (RecordWriter(source).read_json("manifest.json") or {}).get("params")
    asked = dict(getattr(ctx.config, "params", None) or {})
    ctx.record.manifest(params={**(before if isinstance(before, dict) else {}), **asked})


def resume(record, sets=(), params=(), target=None, monitor=None, paths=()):
    source = record_root(record)
    checkpoint = checkpoint_path(source, None)
    directory = fresh_directory(source, target)
    files = tuple(str(path) for path in paths) or (str(source / "resolved.yaml"),)
    checked = check(paths=files, sets=pairs(sets), params=pairs(params, "param"))
    ctx = prepare(checked, directory, monitor, parent=str(source), paths=files)
    carry_params(source, ctx)
    carry_plugins(source, ctx)
    load_checkpoint(ctx, checkpoint, counters=True, rng=True)
    upto = int(ctx.turn)
    replay(ctx, read_lines(source / "events.jsonl"), upto)
    carry_lines(source, ctx, upto)
    carry_best(source, ctx)
    loop.replan(ctx, None)
    ctx.record.event("resume", turn=upto, source=str(source), checkpoint=checkpoint.name)
    return train_into(ctx)


def report_choice(ctx, source):
    if ctx.train.report == "best" and (source / "checkpoints" / "best.pt").exists():
        return "best"
    return None


def open_record(record, which=None, sets=(), params=(), device=None, data=True):
    source = record_root(record)
    if which is not None and which not in CHECKPOINTS:
        raise KalfaError(f"which must be best, last or final, got {which!r}")
    layer = pairs(sets)
    if device is not None:
        layer.append(("device", device))
    files = (str(source / "resolved.yaml"),)
    checked = check(paths=files, sets=layer, params=pairs(params, "param"))
    if checked.errors:
        raise ProblemsError(checked.problems)
    ctx = prepare(checked, False, None, paths=files, fitted=source, sets=data)
    path = checkpoint_path(source, which or report_choice(ctx, source))
    load_checkpoint(ctx, path, only=["models"], counters=False, rng=False)
    ctx.set_modes(False)
    return ctx


def model_entry(ctx, model=None):
    if model is None:
        entry = ctx.predicts()
        if entry is None:
            raise KalfaError("the run has several models and no train.predicts; name one with model")
        return entry
    if model not in ctx.models:
        raise KalfaError(f"model {model!r} is not in the models section, which has {sorted(ctx.models)}")
    return ctx.models[model]


def predict_frame(ctx, loader, frame=None, entry=None):
    return predict.predictions_table(ctx, "test", loader=loader, frame=frame, entry=entry)


def read_frame(path):
    suffix = Path(path).suffix.lower()
    if suffix in (".parquet", ".pq"):
        return pandas.read_parquet(path)
    if suffix == ".tsv":
        return pandas.read_csv(path, sep="\t")
    return pandas.read_csv(path)


def source_of(data):
    if isinstance(data, pandas.DataFrame):
        return data, data, "_frame"
    path = Path(str(data)).expanduser()
    if not path.exists():
        raise KalfaError(f"{data} does not exist; predict takes a .parquet or .csv table, or what the data class "
                         f"reads, such as a folder of images or a text file")
    if path.is_file() and path.suffix.lower() in TABLES:
        frame = read_frame(path)
        return frame, frame, f"_{path.stem}"
    return None, str(path), f"_{path.stem or path.name}"


def predict_record(record, data=None, model=None, which=None, device=None):
    ctx = open_record(record, which=which, device=device, data=data is None)
    if not getattr(ctx.data, "predictions", True):
        raise KalfaError(f"{record}: the data class {type(ctx.data).__name__} writes no predictions table; kalfa "
                         f"generate draws samples from a record whose config has a generate section")
    source = Path(record)
    entry = None if model is None else model_entry(ctx, model)
    if data is None:
        if "test" not in ctx.loaders():
            raise KalfaError(f"{record}: the run has no test set to predict on; predict on a file with data")
        table = predict.predictions_table(ctx, "test", entry=entry)
        tag = f"_{checkpoint_path(source, which or report_choice(ctx, source)).stem}"
    else:
        frame, given, tag = source_of(data)
        loader = ctx.data.loader_for(given)
        if frame is None:
            frame = getattr(getattr(loader, "dataset", None), "frame", None)
        table = predict_frame(ctx, loader, frame, entry)
    if model is not None:
        tag += f"_{model}"
    if not len(table):
        raise KalfaError(f"{record}: there is nothing to predict; the model gave no outputs")
    path = source / f"predictions{tag}.parquet"
    with atomic(path) as temporary:
        table.to_parquet(temporary, index=False)
    return path, table


def chosen_plots(ctx, only):
    if only is None or only == "all":
        return list(ctx.plots)
    names = [name for name in only.split(",") if name] if isinstance(only, str) else list(only)
    unknown = [name for name in names if name not in ctx.plots]
    if unknown:
        raise KalfaError(f"plots {unknown} are not in the plots section, which has {sorted(ctx.plots)}")
    return names


def draw_plots(ctx, record, only=None):
    names = chosen_plots(ctx, only)
    plots, writer = ctx.plots, ctx.record
    ctx.plots = {name: plots[name] for name in names}
    if ctx.record is None:
        ctx.record = RecordWriter(record)
    try:
        return loop.draw_plots(ctx)
    finally:
        ctx.plots, ctx.record = plots, writer
