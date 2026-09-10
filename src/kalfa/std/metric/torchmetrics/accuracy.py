from kalfa.registration import lego
from kalfa.std.metric.torchmetrics.base import ClassMetric


@lego("/metric/torchmetrics/accuracy", state=True, alias="accuracy",
      description="Accuracy of class logits (argmax) against integer labels")
def accuracy():
    return ClassMetric("accuracy", "micro")
