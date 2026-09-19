import logging
import math
from pathlib import Path

from kalfa.std.checkpoint.base import load, load_into, payload, save
from kalfa.std.common.device import Device
from kalfa.std.common.log import logger_for, number


logger_models = logger_for("models")
logger_optimizers = logger_for("optimizers")
logger_training = logger_for("training")
logger_checkpoint = logger_for("training.ckpt")
logger_after = logger_for("after")


def model_line(name, model):
    if not model.initialized:
        return f"{name}: lazy, built on the first batch"
    total = sum(item.numel() for item in model.parameters())
    trainable = sum(item.numel() for item in model.parameters() if item.requires_grad)
    if trainable == total:
        return f"{name}: {total:,} parameters, all trainable"
    return f"{name}: {total:,} parameters, {trainable:,} trainable"


def optimizer_line(name, optimizer):
    lr = optimizer.params.get("lr")
    text = f"{name}: {optimizer.name}" + (f" lr {lr}" if lr is not None else "")
    models = ", ".join(optimizer.models)
    if models:
        text += f" over {models}"
    if optimizer.loss is not None:
        text += f", loss {optimizer.loss}"
    return text


def describe_state(state, total, left, steps):
    if not logger_training.isEnabledFor(logging.INFO):
        return
    for name, model in (state.get("models") or {}).items():
        logger_models.info(model_line(name, model))
    for name, optimizer in (state.get("optimizers") or {}).items():
        logger_optimizers.info(optimizer_line(name, optimizer))
    if steps is not None:
        logger_training.info(f"{steps['total']} steps, {steps['turn']} per turn, {left} turns left")
    elif left != total:
        logger_training.info(f"{total} turns, {left} left")
    else:
        logger_training.info(f"{total} turns")


def checkpoint_line(policy, tags, metrics):
    extra = [tag for tag in tags if tag != "last"]
    if extra:
        logger_checkpoint.info("wrote " + ", ".join(f"{tag}.pt" for tag in tags))
        return
    monitor = policy.monitor
    if monitor is None:
        logger_checkpoint.debug("wrote last.pt")
        return
    value = (metrics or {}).get(monitor)
    if value is None:
        logger_checkpoint.debug(f"wrote last.pt; {monitor} is not in this turn's metrics")
    else:
        logger_checkpoint.debug(f"wrote last.pt; {monitor} {number(value)} is no better than {number(policy.best)}")


def strip_suffix(state, suffix="_next"):
    return {key[:-len(suffix)] if key.endswith(suffix) else key: value for key, value in state.items()}


def init_state(state, epochs, steps, policy=None, resume=None, device=None):
    device = device or Device.cpu()
    device.place(state["models"])
    device.place(state["emas"])
    if resume is not None:
        logger_training.info(f"resuming from {resume}")
        load_into(state["models"], state["optimizers"], state["emas"], state["counters"], state["rules"],
                  load(resume))
        if policy and state["rules"].get("checkpoint") is not None:
            policy.restore(state["rules"]["checkpoint"])
    if epochs is not None:
        total = int(epochs)
    elif steps is not None:
        total = math.ceil(int(steps["total"]) / int(steps["turn"]))
    else:
        raise ValueError("training needs epochs or steps")
    left = max(total - int(state["counters"].get("turn", 0)), 0)
    describe_state(state, total, left, steps)
    return left


def checkpoint(state, policy, metrics=None, record=None):
    if not policy or record is None:
        return None
    parts = strip_suffix(state)
    tags = policy.tags(metrics)
    data = payload(parts.get("models"), parts.get("optimizers"), parts.get("emas"), parts.get("counters"),
                   parts.get("rules"), policy.state())
    for tag in tags:
        save(Path(record) / "checkpoints" / f"{tag}.pt", data)
    checkpoint_line(policy, tags, metrics)
    return None


def save_final(models, optimizers, emas, counters, rules, record=None):
    if record is None:
        return None
    save(Path(record) / "final" / "state.pt", payload(models, optimizers, emas, counters, rules))
    logger_after.debug("final/state.pt written")
    return None


def select(models, emas, which, record=None):
    emas = dict(emas or {})
    if which == "best":
        path = Path(record or ".") / "checkpoints" / "best.pt"
        if not path.exists():
            raise FileNotFoundError(f"report: best needs {path}, but the best checkpoint was never written")
        data = load(path)
        logger_after.info(f"report best: the checkpoint of turn {data.get('turn')}")
        for name, state in data.get("models", {}).items():
            if name in models:
                models[name].load_state_dict(state)
        for name, state in data.get("emas", {}).items():
            if name in emas:
                emas[name].load_state_dict(state)
    elif which != "last":
        raise ValueError(f"report must be best or last, got {which!r}")
    else:
        logger_after.info("report last: the models as training left them")
    selected = dict(models)
    for name, ema in emas.items():
        selected[f"{name}.ema"] = ema
    for model in selected.values():
        model.eval()
    return selected
