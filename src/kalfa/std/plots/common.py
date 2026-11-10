import math
from pathlib import Path


LEGEND_LIMIT = 12
GRID = "#e3e2dd"
INK = "#0b0b0b"
MUTED = "#8c8b86"


def directory_of(record):
    return Path(getattr(record, "directory", record))


def quiet(read, directory):
    try:
        return read(directory)
    except (OSError, ValueError):
        return None


def color_of(figures, position):
    colors = getattr(figures, "colors", None) or tuple(f"C{index}" for index in range(10))
    return colors[position % len(colors)]


def dress(axis):
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def spans_decades(values):
    finite = [value for value in values if math.isfinite(value)]
    if not finite or any(value <= 0 for value in finite):
        return False
    return max(finite) / min(finite) > 100.0


def label_text(value):
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)
