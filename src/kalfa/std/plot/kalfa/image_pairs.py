from kalfa.registration import lego
from kalfa.std.common import figure
from kalfa.std.common.figure import image_tile
from kalfa.std.plot.base import report_loader


@lego("/plot/kalfa/image_pairs", partial=True, alias="image_pairs",
      description="n inputs of the report set next to the predicts model's outputs (reconstructions)")
def image_pairs(predictions, history, models, record, loaders=None, predicts=None, n=8, set=None, name=None):
    import torch

    from kalfa.std.common.runtime import call_model, named_outputs, resolve_model

    set_name, loader = report_loader(loaders, set)
    if loader is None or predicts is None:
        return None
    model = resolve_model(predicts, models)
    model.eval()
    count = min(int(n), len(loader.dataset))
    items = [loader.dataset[position] for position in range(count)]
    batch = {key: torch.stack([item[key] for item in items]) for key in items[0]}
    device = next(iter(model.parameters()), torch.zeros(1)).device
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.no_grad():
        outputs = named_outputs(model, call_model(model, batch))
    wire = next(iter(outputs))
    inputs = batch[model.inputs[0]]
    drawing, axes = figure.sized(1.6 * count, 3.4, 2, count)
    for position in range(count):
        image_tile(axes[0][position], inputs[position])
        image_tile(axes[1][position], outputs[wire][position])
    axes[0][0].set_ylabel(set_name)
    axes[1][0].set_ylabel(wire)
    figure.save(drawing, record, name or "image_pairs")
    return None
