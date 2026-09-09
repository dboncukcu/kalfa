"""The std turn: alternating optimizers over the train loader, one turn per call (an epoch, or K steps)."""

import functools
import logging
import warnings

import torch

from ..registration import lego
from .log import logger
from .runtime import (Context, active_entries, amp_context, amp_scaler, collect_results, entry_loss, loss_scalar,
                      observe_all, resolve_entries, set_modes, to_device, tracker_for, turn_generator)


LOG = logger("training.turn")


def _log_steps(turn, taken, order, stepped_by, accumulate, grad_clip, amp):
    if not LOG.isEnabledFor(logging.DEBUG):
        return
    parts = [f"turn {turn}: {taken} steps"]
    if order:
        parts.append(", ".join(f"{name} x{stepped_by[name]}" for name in order))
    if accumulate > 1:
        parts.append(f"accumulate {accumulate}")
    if grad_clip is not None:
        parts.append(f"grad_clip {grad_clip}")
    if amp:
        parts.append("amp")
    LOG.debug(", ".join(parts))


def effective_loss(name, effects, optimizers):
    """The loss an optimizer minimizes this turn: the rule effects win over the optimizer's own loss."""
    if f"{name}.loss" in effects:
        return effects[f"{name}.loss"]
    if "loss" in effects and len(optimizers) == 1:
        return effects["loss"]
    loss = getattr(optimizers[name], "loss", None)
    if loss is None:
        raise ValueError(f"optimizer {name!r} names no loss")
    return loss


def with_param(entry, param, value):
    """A losses entry with one param replaced: adapters know how, an objective is a partial rebuilt."""
    if hasattr(entry, "with_param"):
        return entry.with_param(param, value)
    base = getattr(entry, "func", entry)
    kwargs = dict(getattr(entry, "keywords", None) or {})
    kwargs[param] = value
    return functools.partial(base, **kwargs)


def apply_effects(effects, models, optimizers, losses):
    """Apply the rule effects that change objects: loss params, optimizer params, trainable flags.

    Returns the losses table with per param replacements applied.
    """
    table = dict(losses)
    for key, value in effects.items():
        if "." not in key:
            continue
        owner, param = key.split(".", 1)
        if param == "loss":
            continue
        if owner in models and param == "trainable":
            model = models[owner]
            model.kalfa_trainable = bool(value)
            for parameter in model.parameters():
                parameter.requires_grad_(bool(value))
        elif owner in optimizers:
            optimizers[owner].set_param(param, value)
        elif owner in table:
            table[owner] = with_param(table[owner], param, value)
        else:
            raise KeyError(f"rule effect {key!r} names neither a model, an optimizer nor a loss")
    return table


class Cursor:
    """Batches of a loader in order; persistent across turns in the steps mode, where the stream restarts when it
    ends, and one pass in the epochs mode, where it ends."""

    def __init__(self, loader, endless):
        self.loader = loader
        self.endless = endless
        self.iterator = iter(loader)
        self.exhausted = False

    @classmethod
    def of(cls, loader, endless):
        if not endless:
            return cls(loader, False)
        cursor = getattr(loader, "kalfa_cursor", None)
        if cursor is None:
            cursor = cls(loader, True)
            loader.kalfa_cursor = cursor
        return cursor

    def take(self, count):
        """Up to ``count`` batches; fewer when a pass ends in the epochs mode, never fewer in the steps mode."""
        batches = []
        while len(batches) < count and not self.exhausted:
            try:
                batches.append(next(self.iterator))
            except StopIteration:
                if self.endless:
                    self.iterator = iter(self.loader)
                    if not batches and _known_empty(self.loader):
                        raise ValueError("the train loader has no batches") from None
                    continue
                self.exhausted = True
        return batches


def _known_empty(loader):
    try:
        return len(loader) == 0
    except TypeError:
        return False


def _clip(optimizer, grad_clip, scaler):
    if grad_clip is None:
        return
    if scaler is not None and scaler.is_enabled():
        scaler.unscale_(optimizer.torch())
    torch.nn.utils.clip_grad_norm_(optimizer.parameters(), float(grad_clip))


def _step(optimizer, scaler):
    if scaler is not None and scaler.is_enabled():
        scaler.step(optimizer.torch())
        scaler.update()
    else:
        optimizer.step()


def _shift_emas(name, optimizers, emas):
    for model_name in optimizers[name].models:
        ema = emas.get(model_name)
        if ema is not None and hasattr(ema, "shift"):
            ema.shift(optimizers[name].models[model_name])


@lego("/turn/kalfa/alternating", alias=["alternating", "supervised"],
            returns=["models", "optimizers", "emas", "counters", "metrics"],
            mutates=["models", "optimizers", "emas", "counters"], bus=["device", "prep", "record"],
            extras=["amp", "grad_clip", "accumulate"],
            description="One turn: every step each optimizer in order minimizes its loss for its steps; a turn is "
                        "an epoch, or K steps with a stream that lives across turns; losses and metrics are the "
                        "running means of the pass")
def alternating(models, optimizers, emas, counters, composites, effects, loader, params, extra, losses, metrics,
                losses_keys, metrics_keys, predicts, steps, device=None, prep=None, record=None):
    params = dict(params or {})
    extra = dict(extra or {})
    effects = dict(effects or {})
    order = list(params.get("order") or optimizers)
    per_steps = dict(params.get("steps") or {})
    fresh_batch = bool(params.get("fresh_batch", False))
    accumulate = int(extra.get("accumulate", 1) or 1)
    grad_clip = extra.get("grad_clip")
    amp = bool(extra.get("amp", False))
    unknown = [name for name in order if name not in optimizers]
    if unknown:
        raise KeyError(f"order names optimizers {unknown} that are not defined")
    resolve_entries(losses, loader=loader)
    losses = apply_effects(effects, models, optimizers, losses)
    set_modes(models, train=True, composites=composites)
    turn = int(counters.get("turn", 0)) + 1
    targets = list(getattr(loader.dataset, "targets", []))
    losses_keys = dict(losses_keys or {})
    active = {name: effective_loss(name, effects, optimizers) for name in order}
    loss_trackers = {name: tracker_for(name, entry, keys)
                     for name, entry, keys in active_entries(losses, losses_keys, "train", turn)}
    metric_trackers = [tracker_for(name, entry, keys, rescale=True)
                       for name, entry, keys in active_entries(metrics, metrics_keys, "train", turn)]
    observers = [tracker for name, tracker in loss_trackers.items() if name not in active.values()] + metric_trackers
    rng = turn_generator(device)
    scaler = amp_scaler(device, amp)
    limit = int(steps["turn"]) if steps is not None else None
    cursor = Cursor.of(loader, endless=steps is not None)
    taken = 0
    last = None
    stepped_by = {name: 0 for name in order}
    while limit is None or taken < limit:
        shared = None if fresh_batch else cursor.take(accumulate)
        if shared is not None and not shared:
            break
        stepped = False
        for name in order:
            optimizer = optimizers[name]
            loss_name = active[name]
            entry = losses[loss_name]
            keys = losses_keys.get(loss_name)
            for _ in range(int(per_steps.get(name, 1))):
                batches = cursor.take(accumulate) if fresh_batch else shared
                if not batches:
                    break
                optimizer.zero_grad()
                for batch in batches:
                    context = Context(to_device(batch, device), models, composites, emas, predicts, targets,
                                      step=int(counters.get("global_step", 0)), epoch=turn, rng=rng, train=True,
                                      prep=prep, set_name="train", scaler=scaler, losses=losses,
                                      losses_keys=losses_keys, record=record)
                    with amp_context(device, amp):
                        value = entry_loss(entry, context, keys)
                    scalar = loss_scalar(value)
                    if not getattr(scalar, "requires_grad", False):
                        raise ValueError(f"loss {loss_name!r} of optimizer {name!r} carries no gradient; its models "
                                         f"are not trainable or the loss does not depend on them")
                    scaled = scalar / len(batches)
                    if scaler is not None and scaler.is_enabled():
                        scaler.scale(scaled).backward()
                    else:
                        scaled.backward()
                    tracker = loss_trackers.get(loss_name)
                    if tracker is not None:
                        tracker.record(value, context.size)
                    last = context
                _clip(optimizer, grad_clip, scaler)
                _step(optimizer, scaler)
                _shift_emas(name, optimizers, emas)
                stepped = True
                stepped_by[name] += 1
            if cursor.exhausted and not stepped:
                break
        if not stepped:
            break
        counters["global_step"] = int(counters.get("global_step", 0)) + 1
        taken += 1
        if last is not None and observers:
            observe_all(observers, last.detached())
        if cursor.exhausted:
            break
    counters["turn"] = turn
    _log_steps(turn, taken, order, stepped_by, accumulate, grad_clip, amp)
    idle = [name for name in order if stepped_by[name] == 0]
    if idle and taken:
        warnings.warn(f"turn {turn}: optimizers {idle} took no step; the train loader ran out of batches before "
                      f"their place in order {order} (steps {per_steps or 'one each'}); more batches per turn or "
                      f"fewer steps for the optimizers before them")
    return {"models": models, "optimizers": optimizers, "emas": emas, "counters": counters,
            "metrics": collect_results([*loss_trackers.values(), *metric_trackers])}
