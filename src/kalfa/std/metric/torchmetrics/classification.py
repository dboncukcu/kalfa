from kalfa.std.metric.torchmetrics.base import ClassMetric, TorchMetric


def accuracy():
    return ClassMetric("accuracy", "micro")


def f1(average="macro"):
    return ClassMetric("f1", average)


def binary_auroc():
    from torchmetrics.classification import BinaryAUROC

    return TorchMetric(BinaryAUROC, "auroc")


def binary_average_precision():
    from torchmetrics.classification import BinaryAveragePrecision

    return TorchMetric(BinaryAveragePrecision, "average_precision")
