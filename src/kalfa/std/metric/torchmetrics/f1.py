from kalfa.registration import lego
from kalfa.std.metric.torchmetrics.base import ClassMetric


@lego("/metric/torchmetrics/f1", state=True, alias="f1",
      description="Macro F1 of class logits (argmax) against integer labels")
def f1(average="macro"):
    return ClassMetric("f1", average)
