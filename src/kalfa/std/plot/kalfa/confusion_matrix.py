import numpy

from kalfa.registration import lego
from kalfa.std.common import figure


@lego("/plot/kalfa/confusion_matrix", partial=True, alias="confusion_matrix",
      description="Confusion matrix of the decoded test predictions against the target labels, counts and "
                  "row shares in every cell")
def confusion_matrix(predictions, history, models, record, name=None):
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
    drawing, axes = figure.sized(side, side * 0.86)
    axis = axes[0][0]
    drawn = axis.imshow(shares, cmap=figure.sequential(), vmin=0, vmax=1)
    for row in range(len(labels)):
        for column in range(len(labels)):
            axis.text(column, row, f"{matrix[row, column]:,}\n{shares[row, column] * 100:.1f}%",
                      ha="center", va="center", fontsize=9,
                      color=figure.INK if shares[row, column] < 0.55 else "#ffffff")
    axis.set_xticks(range(len(labels)), labels)
    axis.set_yticks(range(len(labels)), labels)
    axis.grid(visible=False)
    figure.colorbar(drawing, drawn, axis, "row share", fraction=0.045)
    figure.label(axis, "Confusion matrix", "predicted", "true", note=f"{len(truth):,} test points")
    figure.save(drawing, record, name or "confusion_matrix")
    return None
