from pathlib import Path

import numpy


class Figure:
    categorical = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    surface = "#fcfcfb"
    ink = "#0b0b0b"
    ink_secondary = "#52514e"
    ink_muted = "#8a8880"
    grid_color = "#e3e2dd"
    sequential_steps = ["#eef5fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    diverging_steps = ["#0d366b", "#256abf", "#6da7ec", "#cde2fb", "#f0efec", "#f6c9c8", "#e78d8c", "#c8302f",
                       "#7d1615"]
    formats = ("png", "pdf", "svg")
    styles = ("kalfa", "none")
    panel_width = 6.4
    panel_height = 4.2

    def __init__(self, format="png", width=None, height=None, dpi=150, style="kalfa"):
        if format not in self.formats:
            raise ValueError(f"figures.format must be one of {list(self.formats)}, got {format!r}")
        if style not in self.styles:
            raise ValueError(f"figures.style must be kalfa or none, got {style!r}")
        for name, value in (("width", width), ("height", height), ("dpi", dpi)):
            if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0):
                raise ValueError(f"figures.{name} must be a positive number, got {value!r}")
        self.format = format
        self.width = width
        self.height = height
        self.dpi = dpi
        self.style = style
        self.applied = False
        self.maps = {}

    def with_size(self, width=None, height=None):
        copy = type(self)(self.format, width or self.width, height or self.height, self.dpi, self.style)
        return copy

    def rc(self):
        return {
            "figure.facecolor": self.surface,
            "axes.facecolor": self.surface,
            "savefig.facecolor": self.surface,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelsize": 10,
            "axes.labelcolor": self.ink_secondary,
            "axes.edgecolor": self.grid_color,
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "axes.grid.axis": "both",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": self.grid_color,
            "grid.linewidth": 0.8,
            "grid.alpha": 1.0,
            "text.color": self.ink,
            "xtick.color": self.ink_muted,
            "ytick.color": self.ink_muted,
            "xtick.labelcolor": self.ink_secondary,
            "ytick.labelcolor": self.ink_secondary,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "figure.autolayout": False,
        }

    def pyplot(self):
        import matplotlib
        from matplotlib import pyplot

        matplotlib.use("Agg")
        if not self.applied:
            if self.style == "kalfa":
                matplotlib.rcParams.update(self.rc())
                matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(color=self.categorical)
                if self.dpi:
                    matplotlib.rcParams["savefig.dpi"] = float(self.dpi)
            self.applied = True
        return pyplot

    def map_of(self, name, steps):
        if name not in self.maps:
            from matplotlib.colors import LinearSegmentedColormap

            self.maps[name] = LinearSegmentedColormap.from_list(name, steps)
        return self.maps[name]

    def sequential(self):
        return self.map_of("kalfa_sequential", self.sequential_steps)

    def diverging(self):
        return self.map_of("kalfa_diverging", self.diverging_steps)

    def width_of(self, default=None):
        return float(self.width or default or self.panel_width)

    def height_of(self, default=None):
        return float(self.height or default or self.panel_height)

    def single(self, width=None, height=None):
        return self.pyplot().subplots(figsize=(self.width_of(width), self.height_of(height)))

    def grid(self, rows, columns, width=None, height=None):
        return self.pyplot().subplots(rows, columns, squeeze=False,
                                      figsize=(self.width_of(width) * columns, self.height_of(height) * rows))

    def sized(self, width, height, rows=1, columns=1):
        return self.pyplot().subplots(rows, columns, squeeze=False, figsize=(width, height))

    def tiles(self, rows, columns, side=1.6):
        return self.sized(side * columns, side * rows, rows, columns)

    def label(self, axis, title=None, xlabel=None, ylabel=None, note=None):
        if title:
            axis.set_title(title, color=self.ink, pad=22 if note else None)
        if xlabel:
            axis.set_xlabel(xlabel)
        if ylabel:
            axis.set_ylabel(ylabel)
        if note:
            axis.text(0.0, 1.005, note, transform=axis.transAxes, ha="left", va="bottom", fontsize=9,
                      color=self.ink_muted)
        axis.set_axisbelow(True)
        return axis

    def title(self, drawing, text):
        drawing.suptitle(text, x=0.008, ha="left", fontsize=14, fontweight="bold", color=self.ink)

    def colorbar(self, drawing, drawn, axis, text=None, fraction=0.04):
        bar = drawing.colorbar(drawn, ax=axis, fraction=fraction, pad=0.02)
        if text:
            bar.set_label(text, color=self.ink_secondary, fontsize=9)
        bar.outline.set_visible(False)
        return bar

    def hexbin(self, axis, x, y, gridsize=60):
        return axis.hexbin(x, y, gridsize=int(gridsize), cmap=self.sequential(), bins="log", mincnt=1, linewidths=0)

    def density(self, drawing, axis, x, y, gridsize=60, text="points (log)"):
        drawn = self.hexbin(axis, x, y, gridsize)
        self.colorbar(drawing, drawn, axis, text, fraction=0.045)
        return drawn

    def profile(self, x, y, bins=60, statistic="median"):
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

    def binned(self, x, y, values, bins=55, min_count=15, trim=0.5):
        x_edges = numpy.linspace(numpy.nanpercentile(x, trim), numpy.nanpercentile(x, 100.0 - trim), int(bins) + 1)
        y_edges = numpy.linspace(numpy.nanpercentile(y, trim), numpy.nanpercentile(y, 100.0 - trim), int(bins) + 1)
        count, _, _ = numpy.histogram2d(x, y, bins=[x_edges, y_edges])
        total, _, _ = numpy.histogram2d(x, y, bins=[x_edges, y_edges], weights=values)
        with numpy.errstate(invalid="ignore", divide="ignore"):
            mean = numpy.where(count >= int(min_count), total / numpy.where(count == 0, 1, count), numpy.nan)
        return x_edges, y_edges, mean

    def symmetric(self, matrix, percentile=98.0):
        limit = numpy.nanpercentile(numpy.abs(matrix), percentile) if numpy.isfinite(matrix).any() else 1.0
        return float(limit) if numpy.isfinite(limit) and limit > 0 else 1.0

    def finite(self, *arrays):
        mask = numpy.ones(len(arrays[0]), dtype=bool)
        for array in arrays:
            mask &= numpy.isfinite(numpy.asarray(array, dtype="float64"))
        return [numpy.asarray(array, dtype="float64")[mask] for array in arrays]

    def target(self, record, name):
        directory = Path(record) / "plots"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / name

    def save(self, drawing, record, name, suffix=None):
        drawing.tight_layout()
        path = self.target(record, f"{name}.{suffix or self.format}")
        drawing.savefig(path, bbox_inches="tight")
        self.pyplot().close(drawing)
        return path

    def image_tile(self, axis, tensor):
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
