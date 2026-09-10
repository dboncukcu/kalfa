from kalfa.registration import lego
from kalfa.std.metric.torchmetrics.base import ClassMetric, TorchMetric


@lego("/metric/torchmetrics/accuracy", state=True, alias="accuracy",
      description="Accuracy of class logits (argmax) against integer labels")
def accuracy():
    return ClassMetric("accuracy", "micro")


@lego("/metric/torchmetrics/f1", state=True, alias="f1",
      description="Macro F1 of class logits (argmax) against integer labels")
def f1(average="macro"):
    return ClassMetric("f1", average)


@lego("/metric/torchmetrics/binary_auroc", state=True, alias="auroc",
      description="Area under the ROC curve of binary scores (torchmetrics)")
def binary_auroc():
    from torchmetrics.classification import BinaryAUROC

    return TorchMetric(BinaryAUROC, "auroc")


@lego("/metric/torchmetrics/binary_average_precision", state=True, alias="average_precision",
      description="Average precision of binary scores (torchmetrics)")
def binary_average_precision():
    from torchmetrics.classification import BinaryAveragePrecision

    return TorchMetric(BinaryAveragePrecision, "average_precision")
