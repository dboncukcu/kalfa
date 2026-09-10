import logging
import warnings

from kalfa.registration import lego
from kalfa.std.common.device import Device
from kalfa.std.common.history import History
from kalfa.std.common.log import logger_for
from kalfa.std.common.runtime import Pass, active_entries, collect_results, observe_all, resolve_entries, set_modes
from kalfa.std.turn.base import Cursor, Settings, apply_effects, effective_loss, update


logger = logger_for("training.turn")


def log_steps(turn, taken, order, stepped_by, settings):
    if not logger.isEnabledFor(logging.DEBUG):
        return
    parts = [f"turn {turn}: {taken} steps"]
    if order:
        parts.append(", ".join(f"{name} x{stepped_by[name]}" for name in order))
    if settings.accumulate > 1:
        parts.append(f"accumulate {settings.accumulate}")
    if settings.grad_clip is not None:
        parts.append(f"grad_clip {settings.grad_clip}")
    if settings.amp:
        parts.append("amp")
    logger.debug(", ".join(parts))


class Arrangement:
    def __init__(self, params, optimizers):
        params = dict(params or {})
        self.order = list(params.get("order") or optimizers)
        self.per_steps = dict(params.get("steps") or {})
        self.fresh_batch = bool(params.get("fresh_batch", False))
        unknown = [name for name in self.order if name not in optimizers]
        if unknown:
            raise KeyError(f"order names optimizers {unknown} that are not defined")
        self.stepped_by = {name: 0 for name in self.order}
        self.line = {}

    def step(self, cursor, scope, active, losses, loss_trackers, counters, settings):
        shared = None if self.fresh_batch else cursor.take(settings.accumulate)
        if shared is not None and not shared:
            return False, None
        stepped = False
        last = None
        self.line = {}
        for name in self.order:
            loss_name = active[name]
            for _ in range(int(self.per_steps.get(name, 1))):
                batches = cursor.take(settings.accumulate) if self.fresh_batch else shared
                if not batches:
                    break
                done = update(name, losses[loss_name], scope.losses_keys.get(loss_name), batches, scope,
                              int(counters.get("global_step", 0)), settings, loss_trackers.get(loss_name))
                last = done.context
                self.line[f"loss/{name}"] = done.loss
                self.line[f"lr/{name}"] = scope.optimizers[name].lr()
                if done.norm is not None:
                    self.line[f"grad_norm/{name}"] = done.norm
                stepped = True
                self.stepped_by[name] += 1
            if cursor.exhausted and not stepped:
                break
        return stepped, last


@lego("/turn/kalfa/alternating", alias=["alternating", "supervised"],
      returns=["models", "optimizers", "emas", "counters", "stream", "metrics"],
      mutates=["models", "optimizers", "emas", "counters"], bus=["device", "prep", "record", "monitor"],
      extras=["amp", "grad_clip", "accumulate"],
      description="One turn: every step each optimizer in order minimizes its loss for its steps; a turn is "
                  "an epoch, or K steps with a stream that lives across turns; losses and metrics are the "
                  "running means of the pass; every update writes a line to steps.jsonl (the loss, the learning "
                  "rate and, under grad_clip, the gradient norm of each optimizer) and reaches the monitor")
def alternating(models, optimizers, emas, counters, composites, effects, loader, params, extra, losses, metrics,
                losses_keys, metrics_keys, predicts, steps, stream=None, device=None, prep=None, record=None,
                monitor=None):
    effects = dict(effects or {})
    settings = Settings.of(extra)
    arrangement = Arrangement(params, optimizers)
    resolve_entries(losses, loader=loader)
    resolve_entries(metrics, loader=loader)
    losses = apply_effects(effects, models, optimizers, losses)
    set_modes(models, train=True, composites=composites)
    turn = int(counters.get("turn", 0)) + 1
    device = device or Device.cpu()
    scope = Pass(models, composites, emas, predicts, list(loader.dataset.targets), epoch=turn, device=device,
                 rng=device.generator(), train=True, prep=prep, set_name="train", scaler=device.scaler(settings.amp),
                 losses=losses, losses_keys=dict(losses_keys or {}), record=record)
    scope.optimizers = optimizers
    active = {name: effective_loss(name, effects, optimizers) for name in arrangement.order}
    loss_trackers = {name: entry.tracker(name, keys)
                     for name, entry, keys in active_entries(losses, scope.losses_keys, "train", turn)}
    metric_trackers = [entry.tracker(name, keys, rescale=True)
                       for name, entry, keys in active_entries(metrics, metrics_keys, "train", turn)]
    observers = [tracker for name, tracker in loss_trackers.items() if name not in active.values()] + metric_trackers
    limit = int(steps["turn"]) if steps is not None else None
    cursor = Cursor.of(stream, loader, endless=steps is not None)
    if monitor is not None:
        monitor.turn_begins(limit if limit is not None else cursor.batches())
    taken = 0
    lines = []
    while limit is None or taken < limit:
        stepped, last = arrangement.step(cursor, scope, active, losses, loss_trackers, counters, settings)
        if not stepped:
            break
        counters["global_step"] = int(counters.get("global_step", 0)) + 1
        taken += 1
        line = {"step": counters["global_step"], "turn": turn, **arrangement.line}
        lines.append(line)
        if monitor is not None:
            monitor.step(line)
        if last is not None and observers:
            observe_all(observers, last.detached())
        if cursor.exhausted:
            break
    counters["turn"] = turn
    if record is not None and lines:
        History.append_steps(record, lines)
    log_steps(turn, taken, arrangement.order, arrangement.stepped_by, settings)
    idle = [name for name in arrangement.order if arrangement.stepped_by[name] == 0]
    if idle and taken:
        warnings.warn(f"turn {turn}: optimizers {idle} took no step; the train loader ran out of batches before "
                      f"their place in order {arrangement.order} (steps {arrangement.per_steps or 'one each'}); "
                      f"more batches per turn or fewer steps for the optimizers before them")
    return {"models": models, "optimizers": optimizers, "emas": emas, "counters": counters,
            "stream": cursor if cursor.endless else None,
            "metrics": collect_results([*loss_trackers.values(), *metric_trackers])}
