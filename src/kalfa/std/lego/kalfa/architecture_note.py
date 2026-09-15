from pathlib import Path

import torch

from kalfa.std.common.files import write_json
from kalfa.std.plot.base import report_loader
from kalfa.std.plot.kalfa.architecture import (graph_layout, parameter_count, traced_shapes, trainable_count,
                                               training_layout)


def layout_note(layout):
    columns = layout.ordered()
    boxes = [{"name": box.name, "kind": box.kind, "lines": box.lines, "column": box.column, "row": box.row,
              "detail": box.detail, "layers": box.layers, "parameters": box.parameters}
             for column in sorted(columns) for box in columns[column]]
    return {"boxes": boxes, "arrows": [list(arrow) for arrow in layout.arrows], "widths": dict(layout.widths)}


def architecture_note(models, loaders, composites=None, prep=None, predicts=None, losses=None, losses_keys=None,
                      optimizers=None, record=None, device=None):
    loader = report_loader(loaders)[1]
    batch = next(iter(loader), None) if loader is not None else None
    features = list(prep.features) if prep is not None else None
    note = {}
    for label, model in {**(models or {}), **(composites or {})}.items():
        if label.endswith(".ema"):
            continue
        where = device.torch if device is not None else next(iter(model.parameters()), torch.zeros(1)).device
        shapes = traced_shapes(model, batch, where) if batch is not None else {}
        layout, last = graph_layout(model, label, shapes, features)
        training_layout(layout, model, label, last, predicts, losses, losses_keys, optimizers)
        note[label] = {**layout_note(layout), "parameters": parameter_count(model), "trainable": trainable_count(model)}
    if record is not None:
        write_json(Path(record) / "architecture.json",
                   {"models": note, "features": features or [], "targets": prep.targets if prep is not None else {}})
    return None
