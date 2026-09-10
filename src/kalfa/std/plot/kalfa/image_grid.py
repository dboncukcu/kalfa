from kalfa.registration import lego
from kalfa.std.common import figure
from kalfa.std.common.figure import image_tile
from kalfa.std.plot.base import report_loader
import torch
from kalfa.std.common.runtime import call_model, named_outputs, resolve_model


@lego("/plot/kalfa/image_grid", partial=True, alias="image_grid",
      description="n outputs of the predicts model on the report set as an image grid")
def image_grid(predictions, history, models, record, loaders=None, predicts=None, n=16, set=None, name=None):
    set_name, loader = report_loader(loaders, set)
    if loader is None or predicts is None:
        return None
    model = resolve_model(predicts, models)
    model.eval()
    count = min(int(n), len(loader.dataset))
    if count == 0:
        return None
    items = [loader.dataset[position] for position in range(count)]
    batch = {key: torch.stack([item[key] for item in items]) for key in items[0]}
    device = next(iter(model.parameters()), torch.zeros(1)).device
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.no_grad():
        outputs = named_outputs(model, call_model(model, batch))
    images = outputs[next(iter(outputs))]
    if images.ndim != 4:
        return None
    columns = min(8, count)
    rows = (count + columns - 1) // columns
    drawing, axes = figure.tiles(rows, columns)
    for position in range(rows * columns):
        axis = axes[position // columns][position % columns]
        if position < count:
            image_tile(axis, images[position])
        else:
            axis.axis("off")
    axes[0][0].set_ylabel(set_name)
    figure.save(drawing, record, name or "image_grid")
    return None
