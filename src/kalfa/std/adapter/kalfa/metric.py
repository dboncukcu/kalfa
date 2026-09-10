import copy
import math

from kalfa.registration import lego
from kalfa.std.adapter.base import observed, wires
from kalfa.std.common.runtime import Loss, Tracker, parameter_names
from kalfa.std.metric.base import as_float


class MetricTracker(Tracker):
    def __init__(self, adapter, name, keys=None, rescale=False):
        self.adapter = adapter
        self.name = name
        self.output, self.target = wires(keys)
        self.rescale = rescale
        self.live = copy.deepcopy(adapter.metric)
        self.live.reset()
        self.seen = 0

    def observe(self, context):
        names = parameter_names(self.live.update)
        arguments = {}
        if "predictions" in names or "targets" in names:
            if self.rescale:
                predictions, targets = context.rescaled(self.output, self.target)
            else:
                predictions, targets = context.predictions(self.output), context.target(self.target, self.output)
            if "predictions" in names:
                arguments["predictions"] = observed(predictions)
            if "targets" in names:
                arguments["targets"] = observed(targets)
        scope = context.scope
        if "models" in names:
            arguments["models"] = scope.everything()
        if "batch" in names:
            arguments["batch"] = context.batch
        for name, value in (("rng", scope.rng), ("predicts", scope.predicts), ("record", scope.record),
                            ("turn", scope.epoch), ("prep", scope.prep), ("set", scope.set_name)):
            if name in names:
                arguments[name] = value
        self.live.update(**arguments)
        self.seen += context.size

    def record(self, value, size):
        raise ValueError(f"metric {self.name!r} observes batches; it takes no recorded value")

    def result(self):
        if not self.seen:
            return {self.name: math.nan}
        value = self.live.compute()
        if value is None:
            return {}
        return {self.name: as_float(value)}


@lego("/adapter/kalfa/metric", uses=["predicts"],
      description="Feed a metric the predicts model's output wire and the target field named by the "
                  "definition's keys")
class MetricAdapter(Loss):
    def __init__(self, metric):
        self.metric = metric

    def loss(self, context, keys=None):
        raise ValueError("a metric is observed, never minimized; write it under metrics")

    def tracker(self, name, keys=None, rescale=False):
        return MetricTracker(self, name, keys, rescale)

    def with_param(self, name, value):
        raise ValueError(f"a metric takes no rule effect; {name!r} cannot be set on it")
