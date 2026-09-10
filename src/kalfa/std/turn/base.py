from dataclasses import dataclass

import torch

from kalfa.std.common.runtime import Context, loss_scalar


@dataclass
class Settings:
    amp: bool = False
    grad_clip: float | None = None
    accumulate: int = 1

    @classmethod
    def of(cls, extra):
        extra = dict(extra or {})
        return cls(bool(extra.get("amp", False)), extra.get("grad_clip"), int(extra.get("accumulate", 1) or 1))


def effective_loss(name, effects, optimizers):
    if f"{name}.loss" in effects:
        return effects[f"{name}.loss"]
    if "loss" in effects and len(optimizers) == 1:
        return effects["loss"]
    if optimizers[name].loss is None:
        raise ValueError(f"optimizer {name!r} names no loss")
    return optimizers[name].loss


def apply_effects(effects, models, optimizers, losses):
    table = dict(losses)
    for key, value in effects.items():
        if "." not in key:
            continue
        owner, param = key.split(".", 1)
        if param == "loss":
            continue
        if owner in models and param == "trainable":
            model = models[owner]
            model.trainable = bool(value)
            for parameter in model.parameters():
                parameter.requires_grad_(bool(value))
        elif owner in optimizers:
            optimizers[owner].set_param(param, value)
        elif owner in table:
            table[owner] = table[owner].with_param(param, value)
        else:
            raise KeyError(f"rule effect {key!r} names neither a model, an optimizer nor a loss")
    return table


class Cursor:
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

    def known_empty(self):
        try:
            return len(self.loader) == 0
        except TypeError:
            return False

    def take(self, count):
        batches = []
        while len(batches) < count and not self.exhausted:
            try:
                batches.append(next(self.iterator))
            except StopIteration:
                if not self.endless:
                    self.exhausted = True
                    continue
                self.iterator = iter(self.loader)
                if not batches and self.known_empty():
                    raise ValueError("the train loader has no batches") from None
        return batches


def backward(scaled, scaler):
    if scaler is not None and scaler.is_enabled():
        scaler.scale(scaled).backward()
    else:
        scaled.backward()


def clip_gradients(optimizer, grad_clip, scaler):
    if grad_clip is None:
        return
    if scaler is not None and scaler.is_enabled():
        scaler.unscale_(optimizer.torch())
    torch.nn.utils.clip_grad_norm_(optimizer.parameters(), float(grad_clip))


def step_optimizer(optimizer, scaler):
    if scaler is not None and scaler.is_enabled():
        scaler.step(optimizer.torch())
        scaler.update()
    else:
        optimizer.step()


def update(name, entry, keys, batches, scope, step, settings, tracker=None):
    optimizer = scope.optimizers[name]
    optimizer.zero_grad()
    last = None
    for batch in batches:
        context = Context(scope.device.move(batch), scope, step)
        with scope.device.autocast(settings.amp):
            value = entry.loss(context, keys)
        scalar = loss_scalar(value)
        if not isinstance(scalar, torch.Tensor) or not scalar.requires_grad:
            raise ValueError(f"the loss of optimizer {name!r} carries no gradient; its models are not trainable or "
                             f"the loss does not depend on them")
        backward(scalar / len(batches), scope.scaler)
        if tracker is not None:
            tracker.record(value, context.size)
        last = context
    clip_gradients(optimizer, settings.grad_clip, scope.scaler)
    step_optimizer(optimizer, scope.scaler)
    for model_name in optimizer.models:
        if model_name in scope.emas:
            scope.emas[model_name].shift(optimizer.models[model_name])
    return last
