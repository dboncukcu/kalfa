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


class Cursor:
    def __init__(self, loader, endless):
        self.loader = loader
        self.endless = endless
        self.iterator = iter(loader)
        self.exhausted = False

    @classmethod
    def of(cls, stream, loader, endless):
        if not endless:
            return cls(loader, False)
        if isinstance(stream, Cursor) and stream.loader is loader:
            return stream
        return cls(loader, True)

    def batches(self):
        try:
            return len(self.loader)
        except TypeError:
            return None

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
        return None
    if scaler is not None and scaler.is_enabled():
        scaler.unscale_(optimizer.torch())
    return float(torch.nn.utils.clip_grad_norm_(optimizer.parameters(), float(grad_clip)))


def step_optimizer(optimizer, scaler):
    if scaler is not None and scaler.is_enabled():
        scaler.step(optimizer.torch())
        scaler.update()
    else:
        optimizer.step()


@dataclass
class Update:
    context: Context
    loss: float
    norm: float | None = None


def update(name, entry, keys, batches, scope, step, settings, tracker=None):
    optimizer = scope.optimizers[name]
    optimizer.zero_grad()
    last = None
    total = 0.0
    for batch in batches:
        context = Context(scope.device.move(batch), scope, step)
        with scope.device.autocast(settings.amp):
            value = entry.loss(context, keys)
        scalar = loss_scalar(value)
        if not isinstance(scalar, torch.Tensor) or not scalar.requires_grad:
            raise ValueError(f"the loss of optimizer {name!r} carries no gradient; its models are not trainable or "
                             f"the loss does not depend on them")
        backward(scalar / len(batches), scope.scaler)
        total += float(scalar.detach()) / len(batches)
        if tracker is not None:
            tracker.record(value, context.size)
        last = context
    norm = clip_gradients(optimizer, settings.grad_clip, scope.scaler)
    step_optimizer(optimizer, scope.scaler)
    for model_name in optimizer.models:
        if model_name in scope.emas:
            scope.emas[model_name].shift(optimizer.models[model_name])
    return Update(last, total, norm)
