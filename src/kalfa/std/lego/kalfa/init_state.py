import logging
import math

import torch

from kalfa.registration import lego
from kalfa.std.checkpoint.base import load, load_into
from kalfa.std.common.log import logger_for


logger_models = logger_for("models")
logger_optimizers = logger_for("optimizers")
logger_training = logger_for("training")


def model_line(name, model):
    if not getattr(model, "initialized", True):
        return f"{name}: lazy, built on the first batch"
    total = sum(item.numel() for item in model.parameters())
    trainable = sum(item.numel() for item in model.parameters() if item.requires_grad)
    if trainable == total:
        return f"{name}: {total:,} parameters, all trainable"
    return f"{name}: {total:,} parameters, {trainable:,} trainable"


def optimizer_line(name, optimizer):
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
    describe_state(state, total, left, steps)
    return left
