"""The look of the plots: the palette, the figure settings of the config and the helpers the plot legos draw with."""

import numpy

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e3e2dd"

SEQUENTIAL_STEPS = ["#eef5fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING_STEPS = ["#0d366b", "#256abf", "#6da7ec", "#cde2fb", "#f0efec", "#f6c9c8", "#e78d8c", "#c8302f",
                   "#7d1615"]

FORMATS = ("png", "pdf", "svg")
KEYS = ("format", "width", "height", "dpi", "style")

WIDTH = 6.4
HEIGHT = 4.2

RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 10,
    "axes.labelsize": 10,
    "axes.labelcolor": INK_SECONDARY,
    "axes.edgecolor": GRID,
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "axes.grid.axis": "both",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.alpha": 1.0,
    "text.color": INK,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
    "figure.autolayout": False,
}

DEFAULTS = {"format": "png", "width": None, "height": None, "dpi": 150, "style": "kalfa"}

_settings = dict(DEFAULTS)
_applied = False
_maps = {}


def configure(values):
    """The figures section of the config; every call starts from the defaults, so None resets."""
    global _applied

    _settings.clear()
    _settings.update(DEFAULTS)
    for key, value in (values or {}).items():
        if key in KEYS and value is not None:
            _settings[key] = value
    _applied = False
    return dict(_settings)


def settings():
    return dict(_settings)


def pyplot():
    global _applied

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plot

    if not _applied:
        if _settings.get("style", "kalfa") == "kalfa":
            matplotlib.rcParams.update(RC)
            matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(color=CATEGORICAL)
            if _settings.get("dpi"):
                matplotlib.rcParams["savefig.dpi"] = float(_settings["dpi"])
        _applied = True
    return plot


def _map(name, steps):
    if name not in _maps:
        from matplotlib.colors import LinearSegmentedColormap

        _maps[name] = LinearSegmentedColormap.from_list(name, steps)
    return _maps[name]


def sequential():
    return _map("kalfa_sequential", SEQUENTIAL_STEPS)


def diverging():
    return _map("kalfa_diverging", DIVERGING_STEPS)


def width_of(default=WIDTH):
    return float(_settings.get("width") or default)


def height_of(default=HEIGHT):
    return float(_settings.get("height") or default)


def single(width=WIDTH, height=HEIGHT):
    figure, axes = pyplot().subplots(figsize=(width_of(width), height_of(height)))
    return figure, axes


def grid(rows, columns, width=WIDTH, height=HEIGHT):
    figure, axes = pyplot().subplots(rows, columns, squeeze=False,
                                     figsize=(width_of(width) * columns, height_of(height) * rows))
    return figure, axes


def sized(width, height, rows=1, columns=1):
    figure, axes = pyplot().subplots(rows, columns, squeeze=False, figsize=(width, height))
    return figure, axes


def tiles(rows, columns, side=1.6):
    return sized(side * columns, side * rows, rows, columns)


def label(axis, title=None, xlabel=None, ylabel=None, note=None):
    if title:
        axis.set_title(title, color=INK, pad=22 if note else None)
    if xlabel:
        axis.set_xlabel(xlabel)
    if ylabel:
        axis.set_ylabel(ylabel)
    if note:
        axis.text(0.0, 1.005, note, transform=axis.transAxes, ha="left", va="bottom", fontsize=9, color=INK_MUTED)
    axis.set_axisbelow(True)
    return axis


def title(figure, text):
    figure.suptitle(text, x=0.008, ha="left", fontsize=14, fontweight="bold", color=INK)


def colorbar(figure, drawn, axis, text=None, fraction=0.04):
    bar = figure.colorbar(drawn, ax=axis, fraction=fraction, pad=0.02)
    if text:
        bar.set_label(text, color=INK_SECONDARY, fontsize=9)
    bar.outline.set_visible(False)
    return bar


def hexbin(axis, x, y, gridsize=60):
    return axis.hexbin(x, y, gridsize=int(gridsize), cmap=sequential(), bins="log", mincnt=1, linewidths=0)


def density(figure, axis, x, y, gridsize=60, text="points (log)"):
    drawn = hexbin(axis, x, y, gridsize)
    colorbar(figure, drawn, axis, text, fraction=0.045)
    return drawn


def profile(x, y, bins=60, statistic="median"):
    """A profile over equal count bins of x: the bin centers and the statistic of y inside each one."""
    edges = numpy.unique(numpy.quantile(x, numpy.linspace(0.0, 1.0, int(bins) + 1)))
    if len(edges) < 4:
        return numpy.zeros(0), numpy.zeros(0)
    index = numpy.clip(numpy.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    centers = 0.5 * (edges[:-1] + edges[1:])
    values = numpy.full(len(centers), numpy.nan)
    for position in range(len(centers)):
        picked = y[index == position]
        if len(picked):
            values[position] = numpy.median(picked) if statistic == "median" else numpy.mean(picked)
    keep = numpy.isfinite(values)
    return centers[keep], values[keep]


def binned(x, y, values, bins=55, min_count=15, trim=0.5):
    """The mean of ``values`` over a 2d grid of x and y; bins under ``min_count`` come back as NaN."""
    x_edges = numpy.linspace(numpy.nanpercentile(x, trim), numpy.nanpercentile(x, 100.0 - trim), int(bins) + 1)
    y_edges = numpy.linspace(numpy.nanpercentile(y, trim), numpy.nanpercentile(y, 100.0 - trim), int(bins) + 1)
    count, _, _ = numpy.histogram2d(x, y, bins=[x_edges, y_edges])
    total, _, _ = numpy.histogram2d(x, y, bins=[x_edges, y_edges], weights=values)
    with numpy.errstate(invalid="ignore", divide="ignore"):
        mean = numpy.where(count >= int(min_count), total / numpy.where(count == 0, 1, count), numpy.nan)
    return x_edges, y_edges, mean


def symmetric(matrix, percentile=98.0):
    limit = numpy.nanpercentile(numpy.abs(matrix), percentile) if numpy.isfinite(matrix).any() else 1.0
    return float(limit) if numpy.isfinite(limit) and limit > 0 else 1.0


def finite(*arrays):
    mask = numpy.ones(len(arrays[0]), dtype=bool)
    for array in arrays:
        mask &= numpy.isfinite(numpy.asarray(array, dtype="float64"))
    return [numpy.asarray(array, dtype="float64")[mask] for array in arrays]


def target(record, name):
    from pathlib import Path

    directory = Path(record) / "plots"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


def save(figure, record, name, suffix=None):
    figure.tight_layout()
    path = target(record, f"{name}.{suffix or _settings.get('format') or 'png'}")
    figure.savefig(path, bbox_inches="tight")
    pyplot().close(figure)
    return path
