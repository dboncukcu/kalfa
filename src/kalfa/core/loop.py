import contextlib
import copy
import inspect
import itertools
import math
import random
import time

import numpy
import torch
from torch.nn.parameter import is_lazy

from kalfa.core import graph, state
from kalfa.core.errors import RunError
from kalfa.core.history import Means, history_line, prefixed, progress_text
from kalfa.core.predict import batch_size, empty, enter, write_predictions
from kalfa.core.protocols import Step
from kalfa.core.record import now
from kalfa.core.rules import read_path
from kalfa.core.wrap import active, loss_scalar


def autocast(ctx, enabled):
    if ctx.device is None or not enabled:
        return contextlib.nullcontext()
    return ctx.device.autocast(enabled)


def as_number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, torch.Tensor):
        return float(value.detach()) if value.numel() == 1 else None
    if isinstance(value, (int, float, numpy.integer, numpy.floating)):
        return float(value)
    return None


def detached(outputs):
    return {name: value.detach() if isinstance(value, torch.Tensor) else value for name, value in outputs.items()}


def size_of(loader):
    try:
        return len(loader)
    except (TypeError, NotImplementedError):
        return None


def tell(ctx, phase, done=None, total=None, unit=None):
    if ctx.record is not None:
        ctx.record.note(ctx.turn, phase, done, total, unit)


def observers_of(ctx, set_name):
    skip = set()
    if set_name == "train":
        skip = {id(entry.minimizes) for entry in ctx.optimizers.values() if entry.minimizes is not None}
    return [(name, entry) for name, entry in ctx.losses.items()
            if id(entry) not in skip and active(entry, set_name, ctx.turn)]


def metrics_of(ctx, set_name):
    return [(name, entry) for name, entry in ctx.metrics.items() if active(entry, set_name, ctx.turn)]


def computed(metrics):
    found = {}
    for name, metric in metrics:
        value = metric.compute()
        if isinstance(value, dict):
            for term, item in value.items():
                number = as_number(item)
                if number is not None:
                    found[f"{name}/{term}"] = number
            continue
        number = as_number(value)
        if number is not None:
            found[name] = number
    return found


def observe(ctx, outputs, batch, set_name, means, metrics, losses=None):
    size = batch_size(batch)
    if losses is None:
        losses = observers_of(ctx, set_name)
    for name, entry in losses:
        means.record(name, entry(outputs, batch, ctx), size)
    for _, metric in metrics:
        metric.update(outputs, batch, ctx)


class Cursor:
    def __init__(self, loader, endless):
        self.loader = loader
        self.endless = endless
        self.iterator = iter(loader)
        self.exhausted = False

    def take(self):
        if self.exhausted:
            return None
        try:
            return next(self.iterator)
        except StopIteration:
            if not self.endless:
                self.exhausted = True
                return None
            self.iterator = iter(self.loader)
            try:
                return next(self.iterator)
            except StopIteration:
                raise RunError("the train loader has no batches") from None


class TrainTurn:
    def __init__(self, ctx, step):
        self.ctx = ctx
        self.step = step
        ctx.set_modes(True)
        ctx.set_name = "train"
        self.means = Means()
        self.observers = observers_of(ctx, "train")
        self.metrics = metrics_of(ctx, "train")
        for _, metric in self.metrics:
            metric.reset()
        loaders = ctx.loaders()
        if "train" not in loaders:
            raise RunError(f"there is no train set to train on; the sets are {sorted(loaders)}")
        self.loader = loaders["train"]
        self.limit = None
        if ctx.train.updates:
            total = int(ctx.train.updates["total"])
            per_turn = int(ctx.train.updates.get("turn", total))
            self.limit = max(1, min(per_turn, total - ctx.updates))
            if ctx.cursor is None:
                ctx.cursor = Cursor(self.loader, endless=True)
            self.cursor = ctx.cursor
        else:
            self.cursor = Cursor(self.loader, endless=False)
        self.total = self.limit if self.limit is not None else size_of(self.loader)
        self.taken = 0
        self.seen = 0
        self.note()

    def note(self):
        if self.limit is not None:
            tell(self.ctx, "train", self.taken, self.total, "updates")
        else:
            tell(self.ctx, "train", self.seen, self.total, "batches")

    @property
    def exhausted(self):
        return self.cursor.exhausted

    def more(self):
        return not self.exhausted and (self.limit is None or self.taken < self.limit)

    def next_batch(self):
        batch = self.cursor.take()
        if batch is None:
            return None
        self.seen += 1
        self.note()
        return enter(self.ctx, batch)

    def record_losses(self, losses, size):
        for name, value in losses.items():
            self.means.record(self.ctx.optimizers[name].minimizes.name, value, size)

    def record_update(self, losses):
        ctx = self.ctx
        ctx.updates += 1
        self.taken += 1
        line = {"update": ctx.updates, "turn": ctx.turn}
        for name, value in losses.items():
            line[f"loss/{name}"] = float(loss_scalar(value, name).detach())
            line[f"lr/{name}"] = ctx.optimizers[name].rate()
            norm = ctx.last_norms.get(name)
            if norm is not None:
                line[f"grad_norm/{name}"] = float(norm)
        if ctx.record is not None:
            ctx.record.append("updates.jsonl", line)
        self.note()
        for logger in ctx.loggers:
            logger.update(ctx, line)
        return line

    def observe(self, batch):
        if not self.observers and not self.metrics:
            return
        with torch.no_grad():
            outputs = self.step.eval_batch(batch, self.ctx)
            observe(self.ctx, outputs, batch, "train", self.means, self.metrics, self.observers)

    def close(self):
        if not self.seen:
            raise RunError("the train loader has no batches")
        found = self.means.result()
        found.update(computed(self.metrics))
        return found


def default_train_turn(step, ctx):
    turn = TrainTurn(ctx, step)
    while turn.more():
        batch = turn.next_batch()
        if batch is None:
            break
        with autocast(ctx, step.amp):
            losses = step.train_batch(batch, ctx)
        turn.record_losses(losses, batch_size(batch))
        if getattr(step, "stepped", True):
            turn.record_update(losses)
        turn.observe(batch)
    return turn.close()


def train_turn(ctx):
    step = ctx.step
    if step is None:
        raise RunError("train.step is not set; the default is supervised")
    if type(step).train_turn is Step.train_turn:
        return default_train_turn(step, ctx)
    return step.train_turn(ctx)


def evaluate(ctx, set_name):
    loader = ctx.loaders().get(set_name)
    if loader is None or empty(loader):
        return {}
    losses = observers_of(ctx, set_name)
    metrics = metrics_of(ctx, set_name)
    if not losses and not metrics:
        return {}
    ctx.set_modes(False)
    ctx.set_name = set_name
    means = Means()
    for _, metric in metrics:
        metric.reset()
    seen = 0
    total = size_of(loader)
    tell(ctx, set_name, seen, total, "batches")
    with torch.no_grad():
        for batch in loader:
            seen += 1
            tell(ctx, set_name, seen, total, "batches")
            batch = enter(ctx, batch)
            outputs = ctx.step.eval_batch(batch, ctx)
            observe(ctx, outputs, batch, set_name, means, metrics, losses)
    if not seen:
        return {}
    found = means.result()
    found.update(computed(metrics))
    return found


def tracked(ctx):
    found = {}
    for path in ctx.train.track:
        value = read_path(ctx, path)
        number = as_number(value)
        if number is not None:
            found[f"track/{path}"] = number
            continue
        if isinstance(value, torch.Tensor):
            value = value.detach().flatten().tolist()
        if isinstance(value, (list, tuple)):
            for position, item in enumerate(value):
                number = as_number(item)
                if number is not None:
                    found[f"track/{path}[{position}]"] = number
    return found


def checkpoint_tags(ctx):
    settings = ctx.train.checkpoint
    if settings is False:
        return []
    settings = settings if isinstance(settings, dict) else {}
    tags = []
    watch = settings.get("watch")
    if watch is not None:
        value = as_number(ctx.latest.get(watch))
        if value is None:
            if ctx.turn == 1:
                ctx.logger.warning(f"checkpoint watch {watch!r} is not among the series of turn 1: "
                                   f"{sorted(ctx.latest)}")
        elif not math.isnan(value):
            mode = settings.get("mode", "min")
            if mode not in ("min", "max"):
                raise RunError(f"train.checkpoint.mode must be min or max, got {mode!r}")
            improved = ctx.best is None or (value > ctx.best if mode == "max" else value < ctx.best)
            if improved:
                ctx.best = value
                tags.append("best")
    if settings.get("last", True) is not False:
        tags.append("last")
    every = settings.get("every")
    if every and ctx.turn % int(every) == 0:
        tags.append(f"turn_{ctx.turn}")
    return tags


def write_checkpoints(ctx):
    tags = checkpoint_tags(ctx)
    if ctx.record is None or not tags:
        return tags
    for tag in tags:
        state.save_checkpoint(ctx, ctx.record.path(f"checkpoints/{tag}.pt"))
    ctx.record.event("checkpoint", turn=ctx.turn, tags=tags)
    return tags


def fire_rules(ctx):
    for rule in ctx.rules:
        if rule.fire(ctx) and rule.name not in ctx.fired:
            ctx.fired.append(rule.name)


def replan(ctx, total):
    current = ctx.train.total_turns()
    if current == total:
        return total
    expect = getattr(ctx.monitor, "expect", None)
    if callable(expect):
        expect(current)
    if ctx.record is not None:
        ctx.record.manifest(turns_planned=current)
    ctx.logger.info(f"the run now plans {current} turns of {ctx.train.unit}")
    return current


def train(ctx):
    total = ctx.train.total_turns()
    ctx.logger.info(f"{total} turns of {ctx.train.unit}, {ctx.turn} done")
    written, calls = None, None
    if ctx.record is not None:
        ctx.record.event("run_start", turn=ctx.turn, total=total)
        written = write_summary(ctx)
        calls = FirstCalls(ctx)
        ctx.first_calls = calls
    try:
        first = True
        while ctx.turn < total and ctx.stopping is None:
            total = one_turn(ctx, total)
            if first and calls is not None and calls.helps(written):
                write_summary(ctx)
            first = False
        ctx.logger.info(f"turn {ctx.turn}: {ctx.stopping or 'done'}")
    finally:
        if calls is not None:
            calls.close()
        for logger in ctx.loggers:
            logger.close(ctx)
    return ctx.history


def one_turn(ctx, total):
    ctx.turn += 1
    started = time.perf_counter()
    ctx.fired = []
    tell(ctx, "train")
    results = {"train": train_turn(ctx)}
    loaders = ctx.loaders()
    for set_name in ctx.train.evaluate:
        loader = loaders.get(set_name)
        if loader is None or empty(loader):
            continue
        results[set_name] = evaluate(ctx, set_name)
    metrics = prefixed(results)
    metrics.update(tracked(ctx))
    ctx.latest = metrics
    fire_rules(ctx)
    total = replan(ctx, total)
    write_checkpoints(ctx)
    seconds = time.perf_counter() - started
    line = history_line(ctx, metrics, seconds)
    ctx.history.append(line)
    for logger in ctx.loggers:
        logger.turn(ctx, line)
    if ctx.record is not None:
        ctx.record.append("history.jsonl", line)
        ctx.record.event("turn", turn=ctx.turn, seconds=round(seconds, 3), rules=list(ctx.fired))
        ctx.record.manifest(updated=now(), turn=ctx.turn)
        fired = f"  rules {', '.join(ctx.fired)}" if ctx.fired else ""
        ctx.record.log(progress_text(line, total) + fired)
    if ctx.monitor is not None:
        ctx.monitor.turn(line)
    if ctx.record is not None and ctx.stopping is None and ctx.record.stop_requested():
        note = ctx.record.stop_note() or {}
        ctx.stop(f"stop.json by {note.get('by') or 'unknown'}")
        ctx.stopped_by = note.get("by") or "stop.json"
    note_stop(ctx)
    return total


def note_stop(ctx):
    if ctx.stopping is None or ctx.record is None or getattr(ctx, "stop_noted", False):
        return
    ctx.record.event("stop", turn=ctx.turn, by=getattr(ctx, "stopped_by", None), reason=ctx.stopping)
    ctx.record.log(f"stop asked in turn {ctx.turn}: {ctx.stopping}")
    ctx.stop_noted = True


def moved(ctx, batch):
    return batch if ctx.device is None else ctx.device.move(batch)


def first_batch(ctx):
    loaders = ctx.loaders()
    for set_name in ("valid", "test", "train", *loaders):
        loader = loaders.get(set_name)
        if loader is None or empty(loader):
            continue
        for batch in loader:
            return moved(ctx, batch)
    return None


def device_streams(ctx):
    where = getattr(ctx.device, "torch", None)
    if not isinstance(where, torch.device) or where.type == "cpu":
        return torch.random.fork_rng(devices=[], device_type="cpu")
    index = where.index if where.index is not None else torch.accelerator.current_device_index()
    return torch.random.fork_rng(devices=[index], device_type=where.type)


@contextlib.contextmanager
def unchanged(ctx):
    rng = ctx.rng
    streams = {} if rng is None else {key: generator.get_state() for key, generator in rng.streams.items()}
    pending = {} if rng is None else dict(rng.pending)
    python = random.getstate()
    legacy = numpy.random.get_state()
    try:
        with device_streams(ctx):
            yield
    finally:
        random.setstate(python)
        numpy.random.set_state(legacy)
        if rng is not None:
            for key, generator in rng.streams.items():
                if key in streams:
                    generator.set_state(streams[key])
                    continue
                generator.manual_seed(rng.seed_for(key.rsplit("@", 1)[0]))
                if key in pending:
                    generator.set_state(pending[key])


def counted(parameters):
    return sum(parameter.numel() for parameter in parameters if not is_lazy(parameter))


def module_tree(module):
    return [{"name": name, "class": type(child).__name__, "parameters": counted(child.parameters()),
             "children": module_tree(child)} for name, child in module.named_children()]


def tensors_of(module):
    return itertools.chain(module.parameters(), module.buffers())


def stand_in(entry):
    module = entry.module
    if not any(is_lazy(tensor) for tensor in tensors_of(module)):
        return entry
    kept = {id(tensor): tensor for tensor in tensors_of(module) if not is_lazy(tensor)}
    double = copy.copy(entry)
    double.module = copy.deepcopy(module, kept)
    return double


def kept_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if type(value) in (list, tuple):
        return type(value)(kept_copy(item) for item in value)
    if type(value) is dict:
        return {key: kept_copy(item) for key, item in value.items()}
    return value


class FirstCalls:
    def __init__(self, ctx):
        self.found = {}
        self.handles = {}
        for name, entry in ctx.models.items():
            module = getattr(entry, "module", None)
            if isinstance(module, torch.nn.Module):
                self.handles[name] = module.register_forward_pre_hook(self.keeper(name), with_kwargs=True)

    def keeper(self, name):
        def keep(module, args, kwargs):
            if name not in self.found:
                self.found[name] = (kept_copy(tuple(args)), kept_copy(dict(kwargs or {})))
            handle = self.handles.pop(name, None)
            if handle is not None:
                handle.remove()
        return keep

    def helps(self, summary):
        if not isinstance(summary, dict):
            return False
        return any(isinstance(entry, dict) and entry.get("graph") is None and name in self.found
                   for name, entry in summary.items())

    def close(self):
        for handle in self.handles.values():
            handle.remove()
        self.handles = {}


def shapes_of(entry, result):
    return {label: list(value.shape) for label, value in entry.name_outputs(result).items()
            if isinstance(value, torch.Tensor)}


def traced_outputs(entry, batch):
    with torch.no_grad():
        outputs = entry.call(batch)
    return {label: list(value.shape) for label, value in outputs.items() if isinstance(value, torch.Tensor)}


def module_dtype(module):
    for tensor in tensors_of(module):
        if tensor.is_floating_point() and not is_lazy(tensor):
            return tensor.dtype
    return None


def aligned(value, dtype):
    if isinstance(value, torch.Tensor):
        return value.to(dtype) if dtype is not None and value.is_floating_point() and value.dtype != dtype else value
    if type(value) in (list, tuple):
        return type(value)(aligned(item, dtype) for item in value)
    if type(value) is dict:
        return {key: aligned(item, dtype) for key, item in value.items()}
    return value


def call_for(entry, call):
    dtype = module_dtype(entry.module)
    args, kwargs = call
    return aligned(args, dtype), aligned(kwargs, dtype)


def called_outputs(entry, call):
    args, kwargs = call_for(entry, call)
    with torch.no_grad():
        return shapes_of(entry, entry.module(*args, **kwargs))


def called_graph(name, entry, call):
    args, kwargs = call_for(entry, call)
    try:
        return graph.trace(entry.module, args, graph.argument_names(entry.module, len(args)),
                           list(getattr(entry, "outputs", None) or [name]), keywords=kwargs)
    except Exception:
        return None


def traced_graph(name, entry, batch):
    if batch is None:
        return None
    fields = list(getattr(entry, "inputs", None) or [])
    if not isinstance(batch, dict) or any(field not in batch for field in fields):
        return None
    try:
        return graph.trace(entry.module, [batch[field] for field in fields], fields,
                           list(getattr(entry, "outputs", None) or [name]))
    except Exception:
        return None


def summary_of(name, entry, batch, call=None):
    outputs, found, note, source = {}, None, None, "first batch"
    try:
        used = stand_in(entry)
    except Exception as problem:
        used = entry
        note = f"the model has lazy layers and could not be copied to run it: {type(problem).__name__}: {problem}"
    if note is None and batch is not None:
        try:
            outputs = traced_outputs(used, batch)
        except Exception as problem:
            note = f"the model could not run on the first batch: {type(problem).__name__}: {problem}"
        found = traced_graph(name, used, batch)
    if found is None and call is not None and used is entry:
        found = called_graph(name, entry, call)
        if found is not None:
            source = "first training call"
            if not outputs:
                try:
                    outputs = called_outputs(entry, call)
                except Exception:
                    outputs = {}
    if note is None and batch is None:
        note = "there was no batch to run the model on"
    elif note is None:
        note = "the model ran on the first batch but could not be traced"
    module = used.module
    summary = {"class": graph.qualified(module), "parameters": counted(module.parameters()),
               "trainable": counted(parameter for parameter in module.parameters() if parameter.requires_grad),
               "tree": module_tree(module), "outputs": outputs, "graph": found}
    if found is None:
        summary["graph_note"] = note
    else:
        summary["graph_from"] = source
    return summary


def failed_summary(entry, problem):
    return {"class": graph.qualified(entry.module), "parameters": None, "trainable": None, "tree": [], "outputs": {},
            "graph": None, "graph_note": f"the summary of this model failed: {type(problem).__name__}: {problem}"}


def model_summary(ctx):
    ctx.loaders()
    summary = {}
    with unchanged(ctx):
        batch = first_batch(ctx)
        ctx.set_modes(False)
        calls = getattr(getattr(ctx, "first_calls", None), "found", None) or {}
        for name, entry in ctx.models.items():
            try:
                summary[name] = summary_of(name, entry, batch, calls.get(name))
            except Exception as problem:
                summary[name] = failed_summary(entry, problem)
    return summary


def write_summary(ctx):
    try:
        summary = model_summary(ctx)
    except Exception as problem:
        ctx.logger.warning(f"the model summary could not be written: {type(problem).__name__}: {problem}")
        return None
    ctx.record.write_json("model_summary.json", summary)
    return summary


def write_samples(ctx):
    sampler = ctx.generate
    if sampler is None or ctx.record is None:
        return None
    from kalfa.std.samples import save_samples

    count = int(getattr(sampler, "n", 16) or 16)
    with torch.no_grad():
        samples = sampler.sample(ctx, count)
    return save_samples(samples, ctx.record.path("samples"), "samples")


def takes_context(function):
    try:
        return "ctx" in inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False


def draw_plots(ctx):
    if ctx.record is None:
        return []
    from kalfa.std.figure import Figures

    figures = Figures(ctx.record.path("plots"))
    drawn = []
    for name, plot in ctx.plots.items():
        function, params = plot if isinstance(plot, (tuple, list)) else (plot, {})
        params = dict(params or {})
        if takes_context(function):
            params["ctx"] = ctx
        figure = function(ctx.record.directory, figures, **params)
        if figure is not None:
            figures.save(figure, name)
        drawn.append(name)
    return drawn


def after(ctx):
    record = ctx.record
    tell(ctx, "after")
    if record is not None:
        state.save_checkpoint(ctx, record.path("checkpoints/final.pt"))
        write_summary(ctx)
    if ctx.train.report == "best":
        path = None if record is None else record.path("checkpoints/best.pt")
        if path is None or not path.exists():
            raise RunError("report: best needs checkpoints/best.pt, which is written when train.checkpoint names a "
                           "watch series that improves; write checkpoint: {watch: <series>} or report: last")
        state.load_checkpoint(ctx, path, only=["models"], counters=False, rng=False)
    ctx.set_modes(False)
    tables = {}
    loaders = ctx.loaders() if getattr(ctx.data, "predictions", True) else {}
    tell(ctx, "predictions")
    if "test" in loaders:
        tables["test"] = write_predictions(ctx, "test")
    for set_name in loaders:
        if set_name not in ("train", "valid", "test"):
            tables[set_name] = write_predictions(ctx, set_name)
    tell(ctx, "samples")
    write_samples(ctx)
    tell(ctx, "plots")
    draw_plots(ctx)
    if record is not None:
        record.event("run_end", turn=ctx.turn, stopping=ctx.stopping)
        record.manifest(status="done", finished=now(), turn=ctx.turn)
    return tables
