from pathlib import Path

import torch

from kalfa.std.common.figure import Figure
from kalfa.std.common.files import atomic


def write_samples(samples, target):
    write_sample_files(samples, target, "samples", "grid")


def write_turn_samples(samples, target, turn):
    stem = f"turn_{int(turn):04d}"
    write_sample_files(samples, target, stem, stem)


def write_sample_files(samples, target, stem, image_stem):
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    if isinstance(samples, str):
        (target / f"{stem}.txt").write_text(samples)
        return
    torch.save(samples, target / f"{stem}.pt")
    if isinstance(samples, torch.Tensor) and samples.ndim == 4:
        write_grid(samples, target / f"{image_stem}.png")


def write_grid(images, path, figures=None):
    figures = figures or Figure()
    count = len(images)
    columns = min(8, count)
    rows = (count + columns - 1) // columns
    drawing, axes = figures.tiles(rows, columns)
    for position in range(rows * columns):
        axis = axes[position // columns][position % columns]
        if position < count:
            figures.image_tile(axis, images[position])
        else:
            axis.axis("off")
    with atomic(path) as temporary:
        drawing.savefig(temporary, format=Path(path).suffix.lstrip(".") or "png", bbox_inches="tight")
    figures.pyplot().close(drawing)
