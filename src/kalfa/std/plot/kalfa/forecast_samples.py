import numpy

from kalfa.registration import lego
from kalfa.std.common import figure


@lego("/plot/kalfa/forecast_samples", partial=True, alias="forecast_samples",
      description="n sample windows of the test set: the true horizon against the predicted one")
def forecast_samples(predictions, history, models, record, n=6, name=None):
    if predictions is None or len(predictions) == 0:
        return None
    preds = sorted([column for column in predictions.columns if column.startswith("pred_")],
                   key=lambda column: int(column.rsplit("_", 1)[1]) if column.rsplit("_", 1)[1].isdigit() else 0)
    truths = [column for column in predictions.columns
              if not column.startswith(("pred_", "raw_")) and column != "row"]
    if not preds or not truths:
        return None
    count = min(int(n), len(predictions))
    picked = [int(round(position)) for position in numpy.linspace(0, len(predictions) - 1, count)]
    drawing, axes = figure.sized(figure.width_of(8.0), 2.2 * count, count, 1)
    for axis, position in zip(axes[:, 0], picked):
        row = predictions.iloc[position]
        axis.plot([row[column] for column in truths], color=figure.CATEGORICAL[0], label="true")
        axis.plot([row[column] for column in preds], color=figure.CATEGORICAL[1], linestyle="--",
                  label="predicted")
        figure.label(axis, f"row {row['row']}", None, None)
    axes[0, 0].legend(loc="upper right")
    figure.save(drawing, record, name or "forecast_samples")
    return None
