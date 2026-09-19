import inspect
from pathlib import Path

from cirak.registry import registry

from kalfa.kinds import names_of
from kalfa.std.common.figure import Figure
from kalfa.std.common.history import History
from kalfa.std.common.log import logger_for


logger = logger_for("after.plots")


def refs_of(definition):
    uri = definition.get("lego")
    return registry.facts(uri).refs if isinstance(uri, str) else {}


def needs_of(definition):
    uri = definition.get("lego")
    return names_of(registry.facts(uri).get("needs")) if isinstance(uri, str) else []


def plot_inputs(definition, predictions, history, models):
    refs = refs_of(definition)
    resolved = {}
    for param, name in (definition.get("inputs") or {}).items():
        ref_type = refs.get(param)
        if ref_type == "history":
            resolved[param] = [line.get(name) for line in history or []]
        elif ref_type == "field":
            resolved[param] = predictions[name] if predictions is not None and name in predictions else None
        elif ref_type == "model":
            resolved[param] = models.get(name) if models else None
        else:
            resolved[param] = name
    return resolved


def accepts(plot, name):
    try:
        parameters = inspect.signature(plot).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters or any(parameter.kind is parameter.VAR_KEYWORD for parameter in parameters.values())


def draw(name, plot, definition, predictions, history, everything, predicts, figures, bus, loaders, record, suffix):
    absent = [key for key in needs_of(definition) if bus.get(key) is None]
    if absent:
        logger.info(f"{name} skipped: the bus has no {', '.join(absent)}")
        return
    extra = plot_inputs(definition, predictions, history, everything)
    for key, value in bus.items():
        if accepts(plot, key):
            extra[key] = value
    offered = {"loaders": loaders, "predicts": predicts, "sets": definition.get("sets"), "name": f"{name}{suffix}",
               "figures": figures.with_size(definition.get("width"), definition.get("height"))}
    for key, value in offered.items():
        if accepts(plot, key):
            extra[key] = value
    plot(predictions=predictions, history=history, models=everything, record=record, **extra)


def run_all(predictions, history, models, plots, keys=None, predicts=None, bus=None, record=None, figures=None,
            suffix=""):
    keys = keys or {}
    bus = dict(bus or {})
    figures = figures or Figure()
    if record is not None and (Path(record) / "history.jsonl").exists():
        history = History.read(record).lines
    everything = {**dict(bus.get("composites") or {}), **dict(models or {})}
    loaders = {name[:-len("_loader")]: value for name, value in bus.items()
               if name.endswith("_loader") and value is not None}
    for name, plot in (plots or {}).items():
        logger.debug(f"drawing {name}")
        draw(name, plot, keys.get(name) or {}, predictions, history, everything, predicts, figures, bus, loaders,
             record, suffix)
    if plots:
        logger.info(f"plots: {', '.join(plots)}")
    return None
