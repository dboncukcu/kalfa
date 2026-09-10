import warnings

import torch

from kalfa.registration import lego
from kalfa.std.common.figure import Figure
from kalfa.std.common.log import logger_for
from kalfa.std.common.optional import load
from kalfa.std.plot.base import report_loader


logger = logger_for("after.plots")


@lego("/plot/torchview/architecture", partial=True, alias="torchview", requires="torchview",
      description="torchview's drawing of every report model the batch feeds, under plots/<name>_<model>.png; it "
                  "needs the graphviz dot binary and runs on the device of the run, so a composite keeps its "
                  "referenced models with it")
def architecture(predictions, history, models, record, loaders=None, device=None, name=None, figures=None):
    figures = figures or Figure()
    torchview = load("torchview", "architecture")
    if torchview is None:
        return None
    loader = report_loader(loaders)[1]
    batch = next(iter(loader), None) if loader is not None else None
    if batch is None:
        logger.debug("torchview: no batch to trace with")
        return None
    stem = name or "architecture"
    for label, model in (models or {}).items():
        wires = list(model.inputs)
        if not wires or any(wire not in batch for wire in wires):
            continue
        where = device.torch if device is not None else next(iter(model.parameters()), torch.zeros(1)).device
        try:
            drawing = torchview.draw_graph(model, input_data=[batch[wire][:2].to(where) for wire in wires],
                                           device=where, graph_name=label, expand_nested=True)
            drawing.visual_graph.render(str(figures.target(record, f"{stem}_{label}")), format="png", cleanup=True)
            logger.debug(f"torchview: drew {label}")
        except Exception as exception:
            cause = f" <- {type(exception.__cause__).__name__}: {exception.__cause__}" if exception.__cause__ else ""
            warnings.warn(f"torchview could not draw {label}: {type(exception).__name__}: {exception}{cause}")
    return None
