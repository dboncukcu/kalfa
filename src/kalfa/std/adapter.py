"""Adapters: give criteria and metrics the predictions of the predicts model and the target field."""

import copy
import functools
import math


from ..registration import lego
from ..kinds import SETS
from .metric import as_float


def wires(keys):
    """The output wire and the target field a definition names; both None when it names nothing."""
    keys = keys or {}
    return keys.get("output"), keys.get("target")


class CriterionAdapter:
    def __init__(self, criterion):
        self.criterion = criterion

    def loss(self, context, keys=None, rescale=False):
        output, target = wires(keys)
        if rescale:
            predictions, targets = context.rescaled(output, target)
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


class MetricAdapter:
    def __init__(self, metric):
        self.metric = metric

    def tracker(self, name, keys=None, rescale=False):
        return MetricTracker(self, name, keys, rescale)


class MetricTracker:
    def __init__(self, adapter, name, keys=None, rescale=False):
        self.adapter = adapter
        self.name = name
        self.output, self.target = wires(keys)
        self.rescale = rescale
        self.live = copy.deepcopy(adapter.metric)
        if hasattr(self.live, "reset"):
            self.live.reset()
        self.seen = 0

    def observe(self, context):
        from .runtime import parameter_names

        names = parameter_names(self.live.update)
        arguments = {}
        if "predictions" in names or "targets" in names:
            if self.rescale:
                predictions, targets = context.rescaled(self.output, self.target)
            else:
                predictions, targets = context.predictions(self.output), context.target(self.target, self.output)
            if "predictions" in names:
                arguments["predictions"] = predictions
            if "targets" in names:
                arguments["targets"] = targets
        if "models" in names:
            arguments["models"] = context.everything()
        if "batch" in names:
            arguments["batch"] = context.batch
        if "rng" in names:
            arguments["rng"] = context.rng
        for name, value in (("predicts", context.predicts), ("record", context.record), ("turn", context.epoch),
                            ("prep", context.prep), ("set", context.set_name)):
            if name in names:
                arguments[name] = value
        self.live.update(**arguments)
        self.seen += context.size

    def result(self):
        if not self.seen:
            return {self.name: math.nan}
        value = self.live.compute()
        if value is None:
            return {}
        return {self.name: as_float(value)}


@lego("/adapter/kalfa/criterion", uses=["predicts"],
            description="Feed a criterion the predicts model's output wire and the target field named by the "
                        "definition's keys")
def criterion(criterion):
    return CriterionAdapter(criterion)


@lego("/adapter/kalfa/metric", uses=["predicts"],
            description="Feed a metric the predicts model's output wire and the target field named by the "
                        "definition's keys")
def metric(metric):
    return MetricAdapter(metric)


def all_sets():
    return list(SETS)
