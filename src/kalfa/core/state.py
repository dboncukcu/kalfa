import logging
from pathlib import Path

import torch

from .errors import KalfaError
from .record import atomic


FORMAT = 1


def has_state(obj):
    getter = getattr(obj, "state_dict", None)
    return callable(getter) and bool(getter())


def stateful_nodes(ctx):
    for name, entry in ctx.models.items():
        yield f"models.{name}", entry
        if entry.ema is not None:
            yield f"models.{name}.ema", entry.ema
    for name, loss in ctx.losses.items():
        if has_state(loss):
            yield f"losses.{name}", loss
    for name, metric in ctx.metrics.items():
        if has_state(metric):
            yield f"metrics.{name}", metric
    for name, optimizer in ctx.optimizers.items():
        yield f"optimizers.{name}", optimizer
    if ctx.step is not None and has_state(ctx.step):
        yield "train.step", ctx.step
    for rule in ctx.rules:
        yield f"rules.{rule.name}", rule


def payload(ctx):
    nodes = {path: obj.state_dict() for path, obj in stateful_nodes(ctx)}
    return {"format": FORMAT, "turn": int(ctx.turn), "update": int(ctx.updates), "nodes": nodes,
            "rng": None if ctx.rng is None else ctx.rng.state_dict(), "patches": len(ctx.patches),
            "best": None if ctx.best is None else float(ctx.best)}


def save_checkpoint(ctx, path):
    data = payload(ctx)
    with atomic(path) as temporary:
        torch.save(data, temporary)
    return Path(path)


def read_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def wanted(path, only):
    if only is None:
        return True
    prefixes = (only,) if isinstance(only, str) else tuple(only)
    return any(path == prefix or path.startswith(prefix.rstrip(".") + ".") for prefix in prefixes)


def load_checkpoint(ctx, path, only=None, counters=True, rng=True):
    data = read_checkpoint(path)
    if not isinstance(data, dict) or data.get("format") != FORMAT or "nodes" not in data:
        raise KalfaError(f"{path} is not a kalfa checkpoint of format {FORMAT}")
    objects = dict(stateful_nodes(ctx))
    logger = getattr(ctx, "logger", None) or logging.getLogger("kalfa.run")
    for node_path, state in (data.get("nodes") or {}).items():
        if not wanted(node_path, only):
            continue
        obj = objects.get(node_path)
        if obj is None:
            logger.debug("checkpoint node %s has no object in this run and is skipped", node_path)
            continue
        obj.load_state_dict(state)
    if counters:
        ctx.turn = int(data.get("turn", ctx.turn))
        ctx.updates = int(data.get("update", ctx.updates))
        if "best" in data:
            ctx.best = data["best"]
    if rng and data.get("rng") is not None and getattr(ctx, "rng", None) is not None:
        ctx.rng.load_state_dict(data["rng"])
    return data


def checkpoint_path(directory, which=None):
    base = Path(directory) / "checkpoints"
    if which == "final":
        path = base / "final.pt"
    elif which == "best":
        path = base / "best.pt"
    elif which in (None, "last"):
        path = base / "last.pt"
        if which is None and not path.exists() and (base / "final.pt").exists():
            path = base / "final.pt"
    else:
        raise KalfaError(f"which must be best, last or final, got {which!r}")
    if not path.exists():
        raise KalfaError(f"{directory} has no {path.relative_to(directory)}")
    return path
