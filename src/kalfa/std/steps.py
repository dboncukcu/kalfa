import torch

from kalfa.core.aliases import register
from kalfa.core.errors import RunError
from kalfa.core.loop import TrainTurn, autocast, batch_size, detached
from kalfa.core.protocols import Step
from kalfa.core.wrap import loss_scalar


class Supervised(Step):
    def __init__(self, order=None, amp=False, grad_clip=None, accumulate=1):
        self.order = None if order is None else list(order)
        self.amp = bool(amp)
        self.grad_clip = None if grad_clip is None else float(grad_clip)
        self.accumulate = int(accumulate or 1)
        self.cached = (None, {})
        self.pending = {}
        self.stepped = True

    def arrangement(self, ctx):
        order = list(self.order or ctx.optimizers)
        unknown = [name for name in order if name not in ctx.optimizers]
        if unknown:
            raise RunError(f"{self.path}.order names optimizers {unknown} that are not defined; the optimizers are "
                           f"{sorted(ctx.optimizers)}")
        return order

    def outputs_for(self, batch, ctx):
        entry = ctx.predicts()
        if entry is None:
            return {}
        return entry.call(batch)

    def minimize(self, entry, scalar):
        if self.accumulate <= 1:
            self.stepped = True
            return entry.minimize(scalar, clip=self.grad_clip)
        count = self.pending.get(entry.name, 0) + 1
        if count == 1:
            entry.zero_grad()
        entry.backward(scalar / self.accumulate)
        if count < self.accumulate:
            self.pending[entry.name] = count
            return None
        self.pending[entry.name] = 0
        norm = entry.clip(self.grad_clip)
        entry.step()
        self.stepped = True
        return norm

    def update(self, entry, outputs, batch, ctx):
        if entry.minimizes is None:
            raise RunError(f"optimizer {entry.name!r} minimizes nothing; write minimizes: \"@losses.<name>\"")
        value = entry.minimizes(outputs, batch, ctx)
        scalar = loss_scalar(value, entry.minimizes.name)
        if not isinstance(scalar, torch.Tensor) or not scalar.requires_grad:
            raise RunError(f"the loss {entry.minimizes.name!r} of optimizer {entry.name!r} carries no gradient; its "
                           f"models are frozen or the loss does not depend on their parameters")
        ctx.last_norms[entry.name] = self.minimize(entry, scalar)
        return value

    def train_batch(self, batch, ctx):
        result = {}
        self.stepped = False
        for name in self.arrangement(ctx):
            outputs = self.outputs_for(batch, ctx)
            self.cached = (ctx.batches, detached(outputs))
            result[name] = self.update(ctx.optimizers[name], outputs, batch, ctx)
        return result

    def eval_batch(self, batch, ctx):
        stamp, outputs = self.cached
        if stamp is not None and stamp == ctx.batches:
            return outputs
        return self.outputs_for(batch, ctx)

    def predict(self, batch, ctx):
        return self.outputs_for(batch, ctx)


class Alternating(Supervised):
    def __init__(self, order=None, repeat=None, fresh_batch=False, amp=False, grad_clip=None, accumulate=1):
        super().__init__(order, amp, grad_clip, accumulate)
        self.repeat = dict(repeat or {})
        self.fresh_batch = bool(fresh_batch)

    def step_once(self, entry, batch, ctx):
        with autocast(ctx, self.amp):
            outputs = self.outputs_for(batch, ctx)
            value = self.update(entry, outputs, batch, ctx)
        self.cached = (ctx.batches, detached(outputs))
        return value

    def train_turn(self, ctx):
        turn = TrainTurn(ctx, self)
        order = self.arrangement(ctx)
        repeat = {name: int(self.repeat.get(name, 1)) for name in order}
        stepped_by = {name: 0 for name in order}
        while turn.more():
            shared = None if self.fresh_batch else turn.next_batch()
            if shared is None and not self.fresh_batch:
                break
            losses = {}
            last = None
            self.stepped = False
            for name in order:
                entry = ctx.optimizers[name]
                for _ in range(repeat[name]):
                    batch = turn.next_batch() if self.fresh_batch else shared
                    if batch is None:
                        break
                    value = self.step_once(entry, batch, ctx)
                    turn.means.record(entry.minimizes.name, value, batch_size(batch))
                    losses[name] = value
                    stepped_by[name] += 1
                    last = batch
            if last is None:
                break
            if self.stepped:
                turn.record_update(losses)
            turn.observe(last)
        idle = [name for name in order if stepped_by[name] == 0]
        if idle and turn.taken:
            ctx.logger.warning(f"turn {ctx.turn}: optimizers {idle} took no step; the train loader ran out of "
                               f"batches before their place in order {order}")
        return turn.close()


register("supervised", "kalfa.std.steps.Supervised", slot="step",
         doc="One forward pass of the predicts model per batch, then every optimizer in order minimizes its loss")
register("alternating", "kalfa.std.steps.Alternating", slot="step",
         doc="Per update round each optimizer in order takes repeat[name] steps, on the shared batch or on a fresh "
             "batch per step")
