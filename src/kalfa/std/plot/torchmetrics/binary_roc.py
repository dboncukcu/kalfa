from kalfa.registration import lego
from kalfa.std.common.figure import Figure
from kalfa.std.plot.torchmetrics.base import binary_curve


@lego("/plot/torchmetrics/binary_roc", partial=True,
      description="ROC curve of the raw test scores against the binary target")
def binary_roc(predictions, history, models, record, name=None, figures=None):
    figures = figures or Figure()
    return binary_curve(figures, predictions, record, "binary_roc", "false positive rate", "true positive rate", name)
