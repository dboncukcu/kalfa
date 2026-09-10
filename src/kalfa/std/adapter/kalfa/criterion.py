import functools
import math

from kalfa.registration import lego
from kalfa.std.adapter.base import observed, wires
from kalfa.std.common.runtime import Loss
from kalfa.std.metric.base import as_float


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
        fn = self.criterion
        kwargs = dict(getattr(fn, "keywords", None) or {})
        base = getattr(fn, "func", fn)
        kwargs[name] = value
        return CriterionAdapter(functools.partial(base, **kwargs))

    def resolve(self, **extra):
        """Build the deferred (kind: data) params of the criterion once the data they need exists."""
        from cirak import Deferred

        fn = self.criterion
        kwargs = dict(getattr(fn, "keywords", None) or {})
        if not any(isinstance(value, Deferred) for value in kwargs.values()):
            return self
        base = getattr(fn, "func", fn)
        built = {key: value.build(**extra) if isinstance(value, Deferred) else value for key, value in kwargs.items()}
        self.criterion = functools.partial(base, **built)
        return self


class MeanTracker:
    """The running mean of a criterion over a pass: the model scale under losses, the original scale under metrics."""

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


@lego("/adapter/kalfa/criterion", uses=["predicts"],
      description="Feed a criterion the predicts model's output wire and the target field named by the "
                  "definition's keys")
def criterion(criterion):
    return CriterionAdapter(criterion)
