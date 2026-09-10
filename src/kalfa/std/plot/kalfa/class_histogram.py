from kalfa.registration import lego
from kalfa.std.common import figure
from kalfa.std.plot.base import scores_and_labels


@lego("/plot/kalfa/class_histogram", partial=True, alias="class_histogram",
      description="Histogram of the raw scores of the test set, one series per target class")
def class_histogram(predictions, history, models, record, bins=40, name=None):
    scores, labels = scores_and_labels(predictions)
    if scores is None:
        return None
    drawing, axis = figure.single(width=8.0, height=5.0)
    classes = sorted(set(labels.tolist()))
    for position, label in enumerate(classes):
        axis.hist(scores[labels == label], bins=bins, alpha=0.55, edgecolor="none",
                  color=figure.CATEGORICAL[position % len(figure.CATEGORICAL)], label=str(label))
    axis.legend(loc="upper right")
    figure.label(axis, "Score by class", "score", "points",
                 note=f"{len(scores):,} test points over {len(classes)} classes")
    figure.save(drawing, record, name or "class_histogram")
    return None
