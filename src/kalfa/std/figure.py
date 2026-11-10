from pathlib import Path

import matplotlib

from kalfa.core.record import atomic


matplotlib.use("Agg")

from matplotlib import pyplot  # noqa: E402


PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")


class Figures:
    colors = PALETTE
    width = 6.4
    height = 4.4

    def __init__(self, directory, format="png", dpi=150):
        self.directory = directory
        self.format = format
        self.dpi = dpi

    def new(self, width=None, height=None):
        size = (float(width or self.width), float(height or self.height))
        return pyplot.figure(figsize=size, layout="constrained")

    def save(self, figure, name):
        path = Path(self.directory) / f"{name}.{self.format}"
        try:
            with atomic(path) as temporary:
                figure.savefig(temporary, format=self.format, dpi=self.dpi)
        finally:
            pyplot.close(figure)
        return path
