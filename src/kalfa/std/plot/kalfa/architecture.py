from kalfa.registration import lego
from kalfa.std.common import figure
from kalfa.std.common.log import logger_for
from kalfa.std.common.optional import load
from kalfa.std.plot.base import report_loader
import warnings
import torch


logger = logger_for("after.plots")


def draw_models(models, loaders, record, stem, device=None):
    torchview = load("torchview", "architecture")
    if torchview is None:
        return
    loader = report_loader(loaders)[1]
    batch = next(iter(loader), None) if loader is not None else None
    if batch is None:
        logger.debug("architecture: no batch to trace with, the text is all there is")
        return
    for label, model in (models or {}).items():
        wires = list(model.inputs)
        if not wires or any(wire not in batch for wire in wires):
            continue
        where = device.torch if device is not None else next(iter(model.parameters()), torch.zeros(1)).device
        try:
            drawing = torchview.draw_graph(model, input_data=[batch[wire][:2].to(where) for wire in wires],
                                 device=where, graph_name=label, expand_nested=True)
            drawing.visual_graph.render(str(figure.target(record, f"{stem}_{label}")), format="png", cleanup=True)
            logger.debug(f"architecture: drew {label}")
        except Exception as exception:
            cause = f" <- {type(exception.__cause__).__name__}: {exception.__cause__}" if exception.__cause__ else ""
            warnings.warn(f"architecture: torchview could not draw {label}: "
                          f"{type(exception).__name__}: {exception}{cause}")


@lego("/plot/kalfa/architecture", partial=True, alias="architecture",
      description="The report models printed as text under plots/architecture.txt, and drawn under "
                  "plots/architecture_<model>.png when torchview and graphviz are installed; the drawing "
                  "runs on the device of the run, so a composite keeps its referenced models with it")
def architecture(predictions, history, models, record, loaders=None, device=None, name=None):
    lines = []
    for label, model in (models or {}).items():
        lines.append(f"== {label}")
        lines.append(repr(model))
        lines.append("")
    stem = name or "architecture"
    figure.target(record, f"{stem}.txt").write_text("\n".join(lines))
    draw_models(models, loaders, record, stem, device)
    return None
