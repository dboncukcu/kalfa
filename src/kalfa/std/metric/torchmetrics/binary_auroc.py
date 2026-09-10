from kalfa.registration import lego
from kalfa.std.metric.torchmetrics.base import TorchMetric


@lego("/metric/torchmetrics/binary_auroc", state=True, alias="auroc",
      description="Area under the ROC curve of binary scores (torchmetrics)")
def binary_auroc():
    from torchmetrics.classification import BinaryAUROC

    return TorchMetric(BinaryAUROC, "auroc")
