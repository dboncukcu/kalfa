from kalfa.registration import lego
from kalfa.std.common import figure
from kalfa.std.common.log import logger_for
from kalfa.std.plot.base import report_loader


logger = logger_for("after.plots")


def draw_models(models, loaders, record, stem, device=None):
    import warnings

    import torch

    try:
        from torchview import draw_graph
    except ImportError:
        logger.warning("architecture: torchview is not installed, so the models are written as text only; "
                       "pip install torchview (and the graphviz dot binary) for the drawing")
        return

    loader = report_loader(loaders)[1]
    batch = next(iter(loader), None) if loader is not None else None
    if batch is None:
        logger.debug("architecture: no batch to trace with, the text is all there is")
        return
    for label, model in (models or {}).items():
        wires = getattr(model, "inputs", None)
        if not wires or any(wire not in batch for wire in wires):
            continue
        where = device if device is not None else next(iter(model.parameters()), torch.zeros(1)).device
        try:
            drawing = draw_graph(model, input_data=[batch[wire][:2].to(where) for wire in wires],
                                 device=where, graph_name=label, expand_nested=True)
            drawing.visual_graph.render(str(figure.target(record, f"{stem}_{label}")), format="png", cleanup=True)
            logger.debug(f"architecture: drew {label}")
        except Exception as exc:
            cause = f" <- {type(exc.__cause__).__name__}: {exc.__cause__}" if exc.__cause__ else ""
            warnings.warn(f"architecture: torchview could not draw {label}: "
                          f"{type(exc).__name__}: {exc}{cause}")


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
