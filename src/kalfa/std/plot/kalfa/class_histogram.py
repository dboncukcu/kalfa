from kalfa.registration import lego
from kalfa.std.common.figure import Figure
from kalfa.std.plot.base import scores_and_labels


@lego("/plot/kalfa/class_histogram", partial=True, alias="class_histogram",
      description="Histogram of the raw scores of the test set, one series per target class")
def class_histogram(predictions, history, models, record, bins=40, name=None, figures=None):
    figures = figures or Figure()
    scores, labels = scores_and_labels(predictions)
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
