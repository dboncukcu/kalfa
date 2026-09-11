import numpy

from kalfa.registration import lego
from kalfa.std.common.figure import Figure
from kalfa.std.plot.base import scores_and_labels


@lego("/plot/kalfa/class_histogram", partial=True, alias="class_histogram", refs={"target": "field"},
      description="Histogram of the raw scores of the test set, one series per target class; output names the "
                  "wire and target the field when the table holds several")
def class_histogram(predictions, history, models, record, output=None, target=None, bins=40, name=None,
                    figures=None):
    figures = figures or Figure()
    scores, labels = scores_and_labels(predictions, output, target)
    if scores is None:
        return None
    drawing, axis = figures.single(width=8.0, height=5.0)
    classes = sorted(set(labels.tolist()))
    for position, label in enumerate(classes):
        axis.hist(scores[labels == label], bins=bins, alpha=0.55, edgecolor="none",
                  color=figures.categorical[position % len(figures.categorical)], label=str(label))
    axis.legend(loc="upper right")
    figures.label(axis, "Score by class", "score", "points",
                 note=f"{len(scores):,} test points over {len(classes)} classes")
    figures.save(drawing, record, name or "class_histogram")
    return None


@lego("/plot/kalfa/confusion_matrix", partial=True, alias="confusion_matrix",
      description="Confusion matrix of the decoded test predictions against the target labels, counts and "
                  "row shares in every cell")
def confusion_matrix(predictions, history, models, record, name=None, figures=None):
    figures = figures or Figure()
    if predictions is None or len(predictions) == 0:
        return None
    preds = [column for column in predictions.columns if column.startswith("pred_")]
    targets = [column for column in predictions.columns
               if not column.startswith(("pred_", "raw_")) and column != "row"]
    if not preds or not targets:
        return None
    from sklearn import metrics

    truth = predictions[targets[0]].astype(str).to_numpy()
    guess = predictions[preds[0]].astype(str).to_numpy()
    labels = sorted(set(truth.tolist()) | set(guess.tolist()))
    matrix = metrics.confusion_matrix(truth, guess, labels=labels)
    shares = matrix / numpy.maximum(matrix.sum(axis=1, keepdims=True), 1)
    side = 1.8 + 0.8 * len(labels)
    drawing, axes = figures.sized(side, side * 0.86)
    axis = axes[0][0]
    drawn = axis.imshow(shares, cmap=figures.sequential(), vmin=0, vmax=1)
    for row in range(len(labels)):
        for column in range(len(labels)):
            axis.text(column, row, f"{matrix[row, column]:,}\n{shares[row, column] * 100:.1f}%",
                      ha="center", va="center", fontsize=9,
                      color=figures.ink if shares[row, column] < 0.55 else "#ffffff")
    axis.set_xticks(range(len(labels)), labels)
    axis.set_yticks(range(len(labels)), labels)
    axis.grid(visible=False)
    figures.colorbar(drawing, drawn, axis, "row share", fraction=0.045)
    figures.label(axis, "Confusion matrix", "predicted", "true", note=f"{len(truth):,} test points")
    figures.save(drawing, record, name or "confusion_matrix")
    return None
