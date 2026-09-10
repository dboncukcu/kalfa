import copy
import math

from kalfa.registration import lego
from kalfa.std.adapter.base import observed, wires
from kalfa.std.metric.base import as_float


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
        from kalfa.std.common.runtime import parameter_names

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


@lego("/adapter/kalfa/metric", uses=["predicts"],
      description="Feed a metric the predicts model's output wire and the target field named by the "
                  "definition's keys")
def metric(metric):
    return MetricAdapter(metric)
