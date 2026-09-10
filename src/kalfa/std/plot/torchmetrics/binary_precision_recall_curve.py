from kalfa.registration import lego
from kalfa.std.common.figure import Figure
from kalfa.std.plot.torchmetrics.base import binary_curve


@lego("/plot/torchmetrics/binary_precision_recall_curve", partial=True,
      description="Precision recall curve of the raw test scores against the binary target")
def binary_precision_recall_curve(predictions, history, models, record, name=None, figures=None):
    figures = figures or Figure()
    return binary_curve(figures, predictions, record, "binary_precision_recall_curve", "recall", "precision",
                        name)
