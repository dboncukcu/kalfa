from kalfa.registration import lego
from kalfa.std.metric.torchmetrics.base import TorchMetric


@lego("/metric/torchmetrics/binary_average_precision", state=True, alias="average_precision",
      description="Average precision of binary scores (torchmetrics)")
def binary_average_precision():
    from torchmetrics.classification import BinaryAveragePrecision

    return TorchMetric(BinaryAveragePrecision, "average_precision")
