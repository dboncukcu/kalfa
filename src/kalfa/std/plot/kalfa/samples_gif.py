from kalfa.registration import lego
from kalfa.std.common.figure import Figure
from kalfa.std.plot.base import turn_files
import warnings


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
