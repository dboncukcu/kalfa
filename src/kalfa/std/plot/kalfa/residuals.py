import numpy

from kalfa.registration import lego
from kalfa.std.common.figure import Figure
from kalfa.std.plot.base import pick_pair


@lego("/plot/kalfa/residuals", partial=True, alias="residuals", refs={"target": "field"},
      description="Three panels of one prediction's residual: the distribution with its bias and sigma, the "
                  "residual against the truth as a density, and the mean and median error over equal count "
                  "bins of the target range")
def residuals(predictions, history, models, record, output=None, target=None, bins=20, gridsize=60, name=None,
              figures=None):
    figures = figures or Figure()
    pred, field = pick_pair(predictions, output, target)
    if pred is None:
        return None
    truth, guess = figures.finite(predictions[field].to_numpy(), predictions[pred].to_numpy())
    if len(truth) < 2:
        return None
    error = guess - truth
    drawing, axes = figures.grid(1, 3, width=5.0, height=4.0)
    left, middle, right = axes[0]

    left.hist(error, bins=140, color=figures.categorical[0], edgecolor="none")
    left.axvline(0.0, color=figures.ink_muted, linewidth=1)
    figures.label(left, "Residual distribution", "predicted - true", "points",
                 note=f"bias {error.mean():+.4f}, sigma {error.std():.4f}")

    figures.density(drawing, middle, truth, error, gridsize)
    middle.axhline(0.0, color=figures.categorical[1], linewidth=2)
    figures.label(middle, "Residual vs truth", f"true {field}", "residual")

    centers, median = figures.profile(truth, numpy.abs(error), bins)
    _, mean = figures.profile(truth, numpy.abs(error), bins, statistic="mean")
    if len(centers):
        right.plot(centers, median, marker="o", color=figures.categorical[0], label="median |error|")
        right.plot(centers, mean, marker="s", color=figures.categorical[1], label="mean |error|")
        right.legend(loc="upper left")
    figures.label(right, "Error across the target range", f"true {field} (equal count bins)", "|residual|")

    figures.title(drawing, f"{field} residuals ({len(truth):,} test points)")
    drawing.tight_layout(rect=(0, 0, 1, 0.94))
    figures.save(drawing, record, name or "residuals")
    return None
