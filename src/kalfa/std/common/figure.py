import numpy
from pathlib import Path


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
settings_table = dict(DEFAULTS)
applied = False
maps = {}


def configure(values):
    global applied

    settings_table.clear()
    settings_table.update(DEFAULTS)
    for key, value in (values or {}).items():
        if key in KEYS and value is not None:
            settings_table[key] = value
    applied = False
    return dict(settings_table)


def settings():
    return dict(settings_table)


def pyplot():
    global applied

    import matplotlib
    from matplotlib import pyplot

    matplotlib.use("Agg")

    if not applied:
        if settings_table.get("style", "kalfa") == "kalfa":
            matplotlib.rcParams.update(RC)
            matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(color=CATEGORICAL)
            if settings_table.get("dpi"):
                matplotlib.rcParams["savefig.dpi"] = float(settings_table["dpi"])
        applied = True
    return pyplot


def map_of(name, steps):
    if name not in maps:
        from matplotlib.colors import LinearSegmentedColormap

        maps[name] = LinearSegmentedColormap.from_list(name, steps)
    return maps[name]


def sequential():
    return map_of("kalfa_sequential", SEQUENTIAL_STEPS)


def diverging():
    return map_of("kalfa_diverging", DIVERGING_STEPS)


def width_of(default=WIDTH):
    return float(settings_table.get("width") or default)


def height_of(default=HEIGHT):
    return float(settings_table.get("height") or default)


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
    directory = Path(record) / "plots"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


def save(figure, record, name, suffix=None):
    figure.tight_layout()
    path = target(record, f"{name}.{suffix or settings_table.get('format') or 'png'}")
    figure.savefig(path, bbox_inches="tight")
    pyplot().close(figure)
    return path


def image_tile(axis, tensor):
    array = tensor.detach().cpu().float().numpy()
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = array.transpose(1, 2, 0)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    low, high = float(array.min()), float(array.max())
    if high > low:
        array = (array - low) / (high - low)
    axis.imshow(numpy.clip(array, 0.0, 1.0), cmap="gray" if array.ndim == 2 else None)
    axis.grid(visible=False)
    axis.set_xticks([])
    axis.set_yticks([])
