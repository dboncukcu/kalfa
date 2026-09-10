import math

import torch

from kalfa.registration import lego
from kalfa.std.adapter.base import built, rebound
from kalfa.std.common.runtime import Loss, LossView, Tracker, parameter_names


class ObjectiveTracker(Tracker):
    def __init__(self, adapter, name):
        self.adapter = adapter
        self.name = name
        self.totals = {}
        self.count = 0

    def observe(self, context):
        self.record(self.adapter.loss(context), context.size)

    def record(self, value, size):
        terms = value if isinstance(value, dict) else {"loss": value}
        for term, item in terms.items():
            number = float(item.detach()) if isinstance(item, torch.Tensor) else float(item)
            self.totals[term] = self.totals.get(term, 0.0) + number * size
        self.count += size

    def result(self):
        if not self.count:
            return {self.name: math.nan}
        return {self.name if term == "loss" else f"{self.name}/{term}": total / self.count
                for term, total in self.totals.items()}


@lego("/adapter/kalfa/objective",
      description="Call an objective with every model of the run and the batch, plus the step, epoch, rng, "
                  "scaler and losses view its signature names")
class ObjectiveAdapter(Loss):
    def __init__(self, objective):
        self.objective = objective

    def arguments(self, context):
        names = parameter_names(self.objective)
        scope = context.scope
        found = {}
        for name, value in (("step", context.step), ("epoch", scope.epoch), ("rng", scope.rng),
                            ("scaler", scope.scaler)):
            if name in names:
                found[name] = value
        if "losses" in names:
            found["losses"] = LossView(context)
        return found

    def loss(self, context, keys=None):
        return self.objective(context.scope.everything(), context.batch, **self.arguments(context))

    def tracker(self, name, keys=None, rescale=False):
        return ObjectiveTracker(self, name)

    def with_param(self, name, value):
        return ObjectiveAdapter(rebound(self.objective, name, value))

    def resolve(self, **available):
        self.objective = built(self.objective, **available)
        return self
