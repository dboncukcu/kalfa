import inspect

from cirak.registry import registry

from kalfa.registration import lego
from kalfa.std.common.figure import Figure
from kalfa.std.common.log import logger_for


logger = logger_for("after.plots")


def uri_of(plot):
    base = getattr(plot, "func", plot)
    for uri in registry.uris():
        entry = registry.lookup(uri)
        if entry is not None and entry.target is base:
            return uri
    return None


def plot_inputs(plot, inputs, predictions, history, models):
    uri = uri_of(plot)
    refs = registry.facts(uri).refs if isinstance(uri, str) else {}
    resolved = {}
    for param, name in (inputs or {}).items():
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


def draw(name, plot, definition, predictions, history, everything, predicts, figures, bus, loaders, record):
    extra = plot_inputs(plot, definition.get("inputs"), predictions, history, everything)
    for key, value in bus.items():
        if accepts(plot, key):
            extra[key] = value
    offered = {"loaders": loaders, "predicts": predicts, "sets": definition.get("sets"), "name": name,
               "figures": figures.with_size(definition.get("width"), definition.get("height"))}
    for key, value in offered.items():
        if accepts(plot, key):
            extra[key] = value
    plot(predictions=predictions, history=history, models=everything, record=record, **extra)


@lego("/lego/kalfa/run_all", returns=None, bus=["record", "figures"],
      description="Run every plot of the plots table with the predictions, the history and the models; keys "
                  "carry the definition level keys (inputs, sets, width, height); bus carries everything else "
                  "the run has (prep, the loaders, the device, the final state) and a plot receives whatever "
                  "its signature names, plus loaders, predicts, sets, name and figures, the look of the run's "
                  "plots sized for the definition")
def run_all(predictions, history, models, plots, keys=None, predicts=None, bus=None, record=None, figures=None):
    keys = keys or {}
    bus = dict(bus or {})
    figures = figures or Figure()
    everything = {**dict(bus.get("composites") or {}), **dict(models or {})}
    loaders = {name[:-len("_loader")]: value for name, value in bus.items() if name.endswith("_loader")}
    for name, plot in (plots or {}).items():
        logger.debug(f"drawing {name}")
        draw(name, plot, keys.get(name) or {}, predictions, history, everything, predicts, figures, bus, loaders,
             record)
    if plots:
        logger.info(f"plots: {', '.join(plots)}")
    return None
