import math

from kalfa.std.adapter.base import built, observed, rebound, wires
from kalfa.std.common.runtime import Loss, Tracker
from kalfa.std.metric.base import as_float


class MeanTracker(Tracker):
    def __init__(self, adapter, name, keys=None, rescale=False):
        self.adapter = adapter
        self.name = name
        self.keys = keys
        self.rescale = rescale
        self.total = 0.0
        self.count = 0

    def observe(self, context):
        self.record(self.adapter.loss(context, self.keys, self.rescale), context.size)

    def record(self, value, size):
        self.total += as_float(value) * size
        self.count += size

    def result(self):
        return {self.name: self.total / self.count if self.count else math.nan}


class CriterionAdapter(Loss):
    def __init__(self, criterion):
        self.criterion = criterion

    def loss(self, context, keys=None, rescale=False):
        output, target = wires(keys)
        if rescale:
            predictions, targets = context.rescaled(output, target)
            predictions, targets = observed(predictions), observed(targets)
        else:
            predictions, targets = context.predictions(output), context.target(target, output)
        return self.criterion(predictions, targets)

    def tracker(self, name, keys=None, rescale=False):
        return MeanTracker(self, name, keys, rescale)

    def with_param(self, name, value):
        return CriterionAdapter(rebound(self.criterion, name, value))

    def resolve(self, **available):
        self.criterion = built(self.criterion, **available)
        return self
