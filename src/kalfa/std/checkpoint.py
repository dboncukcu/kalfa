"""Checkpoints: init (device, resume), the per turn checkpoint with its policy, final/ and the report selection."""

import copy
import logging
import math
import random
from pathlib import Path

import numpy
import torch

from ..registration import lego
from .log import logger_for, number

STATE_KEYS = ("models", "optimizers", "emas", "counters", "rules")

logger_training = logger_for("training")
logger_models = logger_for("models")
logger_optimizers = logger_for("optimizers")
logger_checkpoint = logger_for("training.ckpt")
logger_after = logger_for("after")


def _model_line(name, model):
    if not getattr(model, "initialized", True):
        return f"{name}: lazy, built on the first batch"
    total = sum(item.numel() for item in model.parameters())
    trainable = sum(item.numel() for item in model.parameters() if item.requires_grad)
    if trainable == total:
        return f"{name}: {total:,} parameters, all trainable"
    return f"{name}: {total:,} parameters, {trainable:,} trainable"


def _optimizer_line(name, optimizer):
    kind = getattr(getattr(optimizer, "factory", None), "name", "optimizer")
    lr = (getattr(optimizer, "params", None) or {}).get("lr")
    text = f"{name}: {kind}" + (f" lr {lr}" if lr is not None else "")
    models = ", ".join(getattr(optimizer, "models", None) or {})
    if models:
        text += f" over {models}"
    loss = getattr(optimizer, "loss", None)
    if loss is not None:
        text += f", loss {loss}"
    return text


def _describe(state, total, left, steps):
    if not logger_training.isEnabledFor(logging.INFO):
        return
    for name, model in (state.get("models") or {}).items():
        logger_models.info(_model_line(name, model))
    for name, optimizer in (state.get("optimizers") or {}).items():
        logger_optimizers.info(_optimizer_line(name, optimizer))
    if steps is not None:
        logger_training.info(f"{steps['total']} steps, {steps['turn']} per turn, {left} turns left")
    elif left != total:
        logger_training.info(f"{total} turns, {left} left")
    else:
        logger_training.info(f"{total} turns")


def _checkpoint_line(policy, tags, metrics):
    extra = [tag for tag in tags if tag != "last"]
    if extra:
        logger_checkpoint.info("wrote " + ", ".join(f"{tag}.pt" for tag in tags))
        return
    monitor = getattr(policy, "monitor", None)
    if monitor is None:
        logger_checkpoint.debug("wrote last.pt")
        return
    value = (metrics or {}).get(monitor)
    if value is None:
        logger_checkpoint.debug(f"wrote last.pt; {monitor} is not in this turn's metrics")
    else:
        logger_checkpoint.debug(f"wrote last.pt; {monitor} {number(value)} is no better than {number(policy.best)}")


def rng_states():
    states = {"python": random.getstate(), "numpy": numpy.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    return states


def restore_rng(states):
    if not states:
        return
    if "python" in states:
        random.setstate(states["python"])
    if "numpy" in states:
        numpy.random.set_state(states["numpy"])
    if "torch" in states:
        torch.set_rng_state(states["torch"])
    if states.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["cuda"])


def _state_dicts(items):
    return {name: item.state_dict() for name, item in (items or {}).items()}


def payload(models, optimizers, emas, counters, rules, checkpoint=None):
    return {"models": _state_dicts(models), "optimizers": _state_dicts(optimizers), "emas": _state_dicts(emas),
            "counters": dict(counters or {}), "rules": copy.deepcopy(dict(rules or {})),
            "checkpoint": checkpoint, "rng": rng_states(), "turn": int((counters or {}).get("turn", 0))}


def strip_suffix(state, suffix="_next"):
    return {key[:-len(suffix)] if key.endswith(suffix) else key: value for key, value in state.items()}


def save(path, data):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, target)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def load_into(models, optimizers, emas, counters, rules, data):
    for name, state in data.get("models", {}).items():
        if name in models:
            models[name].load_state_dict(state)
    for name, state in data.get("emas", {}).items():
        if name in emas:
            emas[name].load_state_dict(state)
    for name, state in data.get("optimizers", {}).items():
        if name in optimizers:
            optimizers[name].load_state_dict(state)
    counters.clear()
    counters.update(data.get("counters", {}))
    rules.clear()
    rules.update(copy.deepcopy(data.get("rules", {})))
    if data.get("checkpoint") is not None:
        rules["checkpoint"] = data["checkpoint"]
    restore_rng(data.get("rng"))


class Best:
    def __init__(self, monitor, mode):
        self.monitor = monitor
        self.mode = mode
        self.best = None

    def tags(self, metrics):
        value = (metrics or {}).get(self.monitor)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return ["last"]
        value = float(value)
        improved = self.best is None or (value > self.best if self.mode == "max" else value < self.best)
        if improved:
            self.best = value
            return ["best", "last"]
        return ["last"]

    def state(self):
        return {"best": self.best}

    def restore(self, state):
        if state and "best" in state:
            self.best = state["best"]


class Last:
    def tags(self, metrics):
        return ["last"]

    def state(self):
        return None

    def restore(self, state):
        pass


class Snapshot:
    def __init__(self, every):
        self.every = int(every)
        self.seen = 0

    def tags(self, metrics):
        self.seen += 1
        tags = ["last"]
        if self.seen % self.every == 0:
            tags.append(f"snapshot_{self.seen}")
        return tags

    def state(self):
        return {"seen": self.seen}

    def restore(self, state):
        if state and "seen" in state:
            self.seen = int(state["seen"])


@lego("/checkpoint/kalfa/best", alias="best",
            description="Write best.pt when the monitored value improves and last.pt every turn")
def best(monitor, mode="min"):
    if mode not in ("min", "max"):
        raise ValueError(f"mode must be min or max, got {mode!r}")
    return Best(monitor, mode)


@lego("/checkpoint/kalfa/last", alias="last", description="Write last.pt every turn")
def last():
    return Last()


@lego("/checkpoint/kalfa/snapshot", alias="snapshot",
            description="Write snapshot_<n>.pt every n turns and last.pt every turn")
def snapshot(every):
    return Snapshot(every)


@lego("/lego/kalfa/init_state", returns="epochs_left", mutates=["state"], bus=["resume", "device"],
            description="Move the state to the device, load a checkpoint when resuming, count the turns left")
def init_state(state, epochs, steps, resume=None, device=None):
    target = torch.device(device or "cpu")
    for model in state["models"].values():
        model.to(target)
    for ema in state["emas"].values():
        ema.to(target)
    if resume is not None:
        logger_training.info(f"resuming from {resume}")
        load_into(state["models"], state["optimizers"], state["emas"], state["counters"], state["rules"],
                  load(resume))
    if epochs is not None:
        total = int(epochs)
    elif steps is not None:
        total = math.ceil(int(steps["total"]) / int(steps["turn"]))
    else:
        raise ValueError("training needs epochs or steps")
    left = max(total - int(state["counters"].get("turn", 0)), 0)
    _describe(state, total, left, steps)
    return left


@lego("/lego/kalfa/checkpoint", returns=None, bus=["metrics", "record"],
            description="Write the checkpoint files the policy asks for; nothing without a policy")
def checkpoint(state, policy, metrics=None, record=None):
    if policy is None or record is None:
        return None
    parts = strip_suffix(state)
    stored = (parts.get("rules") or {}).get("checkpoint")
    if stored is not None and not getattr(policy, "restored", False):
        policy.restore(stored)
    policy.restored = True
    tags = policy.tags(metrics)
    data = payload(parts.get("models"), parts.get("optimizers"), parts.get("emas"), parts.get("counters"),
                   parts.get("rules"), policy.state())
    for tag in tags:
        save(Path(record) / "checkpoints" / f"{tag}.pt", data)
    _checkpoint_line(policy, tags, metrics)
    return None


@lego("/lego/kalfa/save_final", returns=None, bus=["record"],
            description="Write final/state.pt with the full state once training ends")
def save_final(models, optimizers, emas, counters, rules, record=None):
    if record is None:
        return None
    save(Path(record) / "final" / "state.pt", payload(models, optimizers, emas, counters, rules))
    logger_after.debug("final/state.pt written")
    return None


@lego("/lego/kalfa/select", returns="selected", bus=["record"],
            description="The report models: copies loaded from best.pt, or the final state for last")
def select(models, emas, which, record=None):
    copies = copy.deepcopy(dict(models))
    ema_copies = copy.deepcopy(dict(emas or {}))
    if which == "best":
        path = Path(record or ".") / "checkpoints" / "best.pt"
        if not path.exists():
            raise FileNotFoundError(f"report: best needs {path}, but the best checkpoint was never written")
        data = load(path)
        logger_after.info(f"report best: the checkpoint of turn {data.get('turn')}")
        for name, state in data.get("models", {}).items():
            if name in copies:
                copies[name].load_state_dict(state)
        for name, state in data.get("emas", {}).items():
            if name in ema_copies:
                ema_copies[name].load_state_dict(state)
    elif which != "last":
        raise ValueError(f"report must be best or last, got {which!r}")
    else:
        logger_after.info("report last: the models as training left them")
    selected = dict(copies)
    for name, ema in ema_copies.items():
        selected[f"{name}.ema"] = ema
    for model in selected.values():
        model.eval()
    return selected
