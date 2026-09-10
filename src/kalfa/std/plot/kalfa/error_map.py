import numpy

from kalfa.registration import lego
from kalfa.std.common.figure import Figure
from kalfa.std.plot.base import first_set, pick_pair, set_frame


def points_text(count):
    return f"{int(count)} point" + ("" if int(count) == 1 else "s")


@lego("/plot/kalfa/error_map", partial=True, alias="error_map",
      refs={"x": "column", "y": "column", "target": "field"},
      description="The error of one prediction over a 2d grid of two columns: with statistic residual blue "
                  "is a prediction below the truth and red above it, with abs the mean absolute error; bins "
                  "holding fewer than min_count points stay empty")
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
