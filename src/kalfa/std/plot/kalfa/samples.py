import warnings

import torch

from kalfa.registration import lego
from kalfa.std.common.figure import Figure
from kalfa.std.plot.base import turn_files


@lego("/plot/kalfa/samples_gif", partial=True, alias="samples_gif",
      description="The per turn sample grids of samples/turn_*.png as an animation; skipped with a warning "
                  "when there are none")
def samples_gif(predictions, history, models, record, name=None, duration=400, figures=None):
    figures = figures or Figure()
    from PIL import Image

    frames = turn_files(record, ".png")
    if not frames:
        warnings.warn("samples_gif: no samples/turn_*.png in the record; add a sample_writer metric")
        return None
    images = []
    for frame in frames:
        with Image.open(frame) as handle:
            images.append(handle.convert("RGB"))
    images[0].save(figures.target(record, f"{name or 'samples_gif'}.gif"), save_all=True,
                   append_images=images[1:], duration=int(duration), loop=0)
    return None


@lego("/plot/kalfa/samples_matrix", partial=True, alias="samples_matrix",
      description="A matrix of the per turn samples of samples/turn_*.pt: one row per turn, n columns; "
                  "skipped with a warning when there are none")
def samples_matrix(predictions, history, models, record, name=None, n=8, figures=None):
    figures = figures or Figure()
    files = turn_files(record, ".pt")
    if not files:
        warnings.warn("samples_matrix: no samples/turn_*.pt in the record; add a sample_writer metric")
        return None
    rows = []
    for path in files:
        samples = torch.load(path, weights_only=False)
        if isinstance(samples, torch.Tensor) and samples.ndim == 4:
            rows.append((path.stem.split("_", 1)[1].lstrip("0") or "0", samples[:int(n)]))
    if not rows:
        warnings.warn("samples_matrix: the turn samples are not images")
        return None
    columns = max(len(samples) for _, samples in rows)
    drawing, axes = figures.tiles(len(rows), columns)
    for row, (turn, samples) in enumerate(rows):
        for column in range(columns):
            axis = axes[row][column]
            if column < len(samples):
                figures.image_tile(axis, samples[column])
            else:
                axis.axis("off")
        axes[row][0].set_ylabel(f"turn {turn}")
    figures.save(drawing, record, name or "samples_matrix")
    return None
