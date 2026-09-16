import numpy

from kalfa.std.common.figure import Figure
from kalfa.std.plot.base import (
    first_set,
    panel_title,
    pick_pair,
    prediction_pairs,
    r2_of,
    set_frame,
    shared_histograms,
)


def points_text(count):
    return f"{int(count)} point" + ("" if int(count) == 1 else "s")


def pred_vs_true(predictions, history, models, record, name=None, columns=4, kind="auto", gridsize=70, figures=None):
    figures = figures or Figure()
    pairs = prediction_pairs(predictions)
    if not pairs:
        return None
    width = max(1, min(int(columns or 4), len(pairs)))
    rows = -(-len(pairs) // width)
    drawing, axes = figures.grid(rows, width, width=5.4, height=4.6)
    panels = [axis for row in axes for axis in row]
    paired = [field for _, field in pairs]
    for axis, (pred, target) in zip(panels, pairs):
        true, guess = figures.finite(predictions[target].to_numpy(), predictions[pred].to_numpy())
        if not len(true):
            axis.axis("off")
            continue
        dense = kind == "hexbin" or (kind == "auto" and len(true) >= 2000)
        if dense:
            figures.density(drawing, axis, true, guess, gridsize)
        else:
            axis.scatter(true, guess, s=12, alpha=0.5, color=figures.categorical[0], edgecolors="none")
        low = float(min(true.min(), guess.min()))
        high = float(max(true.max(), guess.max()))
        axis.plot([low, high], [low, high], color=figures.categorical[1], linewidth=2, label="perfect")
        axis.legend(loc="upper left")
        figures.label(axis, panel_title(pred, target, paired), "true", "predicted",
                     note=f"R2 = {r2_of(true, guess):.4f} on {len(true):,} points")
    for axis in panels[len(pairs):]:
        axis.axis("off")
    figures.save(drawing, record, name or "pred_vs_true")
    return None


def ratio_of(pred, data):
    with numpy.errstate(divide="ignore", invalid="ignore"):
        ratio = numpy.where(data > 0, pred / numpy.where(data > 0, data, 1), numpy.nan)
        spread = numpy.where((data > 0) & (pred > 0), ratio * numpy.sqrt(1.0 / numpy.where(pred > 0, pred, 1)
                                                                       + 1.0 / numpy.where(data > 0, data, 1)), 0.0)
    return ratio, spread


def ratio_window(ratio, spread):
    finite = numpy.isfinite(ratio)
    low = min(0.5, float(numpy.min((ratio - spread)[finite]))) if finite.any() else 0.5
    high = max(1.5, float(numpy.max((ratio + spread)[finite]))) if finite.any() else 1.5
    return max(0.0, low), min(3.0, high)


def pred_histogram(predictions, history, models, record, columns=4, bins=40, log=False, name=None, figures=None):
    figures = figures or Figure()
    pairs = prediction_pairs(predictions)
    if not pairs:
        return None
    width = max(1, min(int(columns or 4), len(pairs)))
    rows = -(-len(pairs) // width)
    drawing, axes = figures.pyplot().subplots(2 * rows, width, squeeze=False,
                                              figsize=(figures.width_of(5.4) * width, figures.height_of(5.4) * rows),
                                              gridspec_kw={"height_ratios": [3.0, 1.0] * rows})
    paired = [field for _, field in pairs]
    panels = [(axes[2 * (position // width)][position % width], axes[2 * (position // width) + 1][position % width])
              for position in range(rows * width)]
    for (top, below), (pred, target) in zip(panels, pairs):
        truth, guess = figures.finite(predictions[target].to_numpy(), predictions[pred].to_numpy())
        if not len(truth):
            top.axis("off")
            below.axis("off")
            continue
        edges, data, counts = shared_histograms(truth, guess, bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        light, dark = figures.sequential_steps[2], figures.sequential_steps[6]
        top.hist(truth, bins=edges, histtype="stepfilled", color=light, alpha=0.55, edgecolor=light, linewidth=1.2,
                 label=target, log=bool(log))
        top.hist(guess, bins=edges, histtype="step", color=dark, linewidth=2, label=pred, log=bool(log))
        top.legend(loc="upper right")
        top.tick_params(labelbottom=False)
        figures.label(top, panel_title(pred, target, paired), None, "points",
                     note=f"{len(truth):,} points over {len(data)} shared bins")
        ratio, spread = ratio_of(counts, data)
        below.axhline(1.0, color=figures.ink_secondary, linewidth=1, linestyle="--")
        below.errorbar(centers, ratio, yerr=spread, fmt="o", markersize=3.5, color=dark, ecolor=dark, elinewidth=1.2,
                       capsize=2)
        below.set_xlim(top.get_xlim())
        below.set_ylim(*ratio_window(ratio, spread))
        figures.label(below, None, target, "pred / true")
    for top, below in panels[len(pairs):]:
        top.axis("off")
        below.axis("off")
    figures.save(drawing, record, name or "pred_histogram")
    return None


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


def error_map(predictions, history, models, record, loaders=None, prep=None, sets=None, x=None, y=None,
              output=None, target=None, statistic="residual", bins=55, min_count=15, name=None, figures=None):
    figures = figures or Figure()
    pred, field = pick_pair(predictions, output, target)
    table = set_frame(loaders, prep, first_set(sets, "test"))
    if pred is None or table is None or x is None or y is None:
        return None
    if x not in table.columns or y not in table.columns or "row" not in predictions.columns:
        return None
    picked = table.reindex(predictions["row"].to_numpy())
    error = predictions[pred].to_numpy(dtype="float64") - predictions[field].to_numpy(dtype="float64")
    if statistic == "abs":
        error = numpy.abs(error)
    across, along, values = figures.finite(picked[x].to_numpy(), picked[y].to_numpy(), error)
    if len(across) < int(min_count):
        return None
    x_edges, y_edges, mean = figures.binned(across, along, values, bins, min_count)
    drawing, axis = figures.single(width=7.0, height=5.0)
    if statistic == "abs":
        drawn = axis.pcolormesh(x_edges, y_edges, mean.T, cmap=figures.sequential(), shading="auto")
        note = f"mean absolute error; bins with at least {points_text(min_count)}"
        text = "mean |error|"
    else:
        limit = figures.symmetric(mean)
        drawn = axis.pcolormesh(x_edges, y_edges, mean.T, cmap=figures.diverging(), vmin=-limit, vmax=limit,
                                shading="auto")
        note = (f"blue: prediction below truth, red: prediction above truth; bins with at least "
                f"{points_text(min_count)}")
        text = "mean residual (pred - true)"
    figures.colorbar(drawing, drawn, axis, text)
    axis.grid(visible=False)
    figures.label(axis, f"Where the model is off, over ({x}, {y})", x, y, note=note)
    figures.save(drawing, record, name or "error_map")
    return None


def forecast_samples(predictions, history, models, record, n=6, name=None, figures=None):
    figures = figures or Figure()
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
    drawing, axes = figures.sized(figures.width_of(8.0), 2.2 * count, count, 1)
    for axis, position in zip(axes[:, 0], picked):
        row = predictions.iloc[position]
        axis.plot([row[column] for column in truths], color=figures.categorical[0], label="true")
        axis.plot([row[column] for column in preds], color=figures.categorical[1], linestyle="--",
                  label="predicted")
        figures.label(axis, f"row {row['row']}", None, None)
    axes[0, 0].legend(loc="upper right")
    figures.save(drawing, record, name or "forecast_samples")
    return None
