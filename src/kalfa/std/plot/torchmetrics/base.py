import numpy

from kalfa.std.common import figure
from kalfa.std.plot.base import scores_and_labels


def area_of(x, y):
    return float(abs(numpy.sum(numpy.diff(x) * (y[:-1] + y[1:]) / 2.0)))


def binary_curve(predictions, record, kind, xlabel, ylabel, name=None):
    import torch
    from torchmetrics.functional.classification import binary_precision_recall_curve, binary_roc

    scores, labels = scores_and_labels(predictions)
    if scores is None or len(set(labels.tolist())) < 2:
        return None
    score = torch.as_tensor(numpy.array(scores, dtype="float32"))
    label = torch.as_tensor(numpy.array(labels)).long()
    if kind == "binary_roc":
        x, y, _ = binary_roc(score, label)
        title, note = "ROC", f"AUC = {area_of(x.numpy(), y.numpy()):.4f}"
    else:
        precision, recall, _ = binary_precision_recall_curve(score, label)
        x, y = recall, precision
        title, note = "Precision and recall", f"AP = {area_of(x.numpy(), y.numpy()):.4f}"
    drawing, axis = figure.single(width=6.0, height=5.0)
    axis.plot(x.numpy(), y.numpy(), color=figure.CATEGORICAL[0], label=note)
    if kind == "binary_roc":
        axis.plot([0, 1], [0, 1], color=figure.INK_MUTED, linewidth=1, linestyle="--", label="chance")
    else:
        axis.axhline(float(label.float().mean()), color=figure.INK_MUTED, linewidth=1, linestyle="--",
                     label="base rate")
    axis.legend(loc="lower right" if kind == "binary_roc" else "lower left")
    figure.label(axis, title, xlabel, ylabel, note=f"{len(scores):,} test points")
    figure.save(drawing, record, name or kind)
    return None
