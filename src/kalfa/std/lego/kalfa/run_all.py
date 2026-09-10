from kalfa.registration import lego
from kalfa.std.common import figure
from kalfa.std.common.log import logger_for
import inspect
from cirak.registry import registry


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


def draw_all(predictions, history, everything, plots, keys, predicts, figures, bus, loaders, record):
    for name, plot in (plots or {}).items():
        logger.debug(f"drawing {name}")
        definition = keys.get(name) or {}
        extra = plot_inputs(plot, definition.get("inputs"), predictions, history, everything)
        for key, value in bus.items():
            if accepts(plot, key):
                extra[key] = value
        if accepts(plot, "loaders"):
            extra["loaders"] = loaders
        if accepts(plot, "predicts"):
            extra["predicts"] = predicts
        if accepts(plot, "sets"):
            extra["sets"] = definition.get("sets")
        if accepts(plot, "name"):
            extra["name"] = name
        size = {key: definition[key] for key in ("width", "height") if definition.get(key) is not None}
        if size:
            figure.configure({**figure.settings(), **size})
        try:
            plot(predictions=predictions, history=history, models=everything, record=record, **extra)
        finally:
            if size:
                figure.configure(figures)
    return None


@lego("/lego/kalfa/run_all", returns=None, bus=["record"],
      description="Run every plot of the plots table with the predictions, the history and the models; keys "
                  "carry the definition level keys (inputs, sets, width, height); bus carries everything else "
                  "the run has (prep, the loaders, the device, the final state) and a plot receives whatever "
                  "its signature names, plus loaders, predicts, sets and name; figures carries the figure "
                  "settings of the config")
def run_all(predictions, history, models, plots, keys=None, predicts=None, figures=None, bus=None, record=None):
    keys = keys or {}
    bus = dict(bus or {})
    before = figure.settings()
    figure.configure(figures)
    everything = {**dict(bus.get("composites") or {}), **dict(models or {})}
    loaders = {name: bus.get(f"{name}_loader") for name in ("train", "valid", "test")}
    try:
        draw_all(predictions, history, everything, plots, keys, predicts, figures, bus, loaders, record)
    finally:
        figure.configure(before)
    if plots:
        logger.info(f"plots: {', '.join(plots)}")
    return None
