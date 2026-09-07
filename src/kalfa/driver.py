"""The driver: the resolved config becomes the cirak document the templates open (CONFIG.md section 8).

Direct transfer for data.*, metrics, losses, plots, generate, training.*, seed, device, record; eleven reshapings,
each reading only the shape of a section (a key's presence, a list's length, a string against a mapping).
"""

from cirak.registry import registry as default_registry

from .config import resolve_alias
from .kinds import kalfa_kind, DEFINITION_KEYS, TRAINING_FIXED

ADAPTER_CRITERION = "/adapter/kalfa/criterion"
ADAPTER_METRIC = "/adapter/kalfa/metric"
RANDOM_SPLIT = "/split/kalfa/random"
BUILDER = "/builder/kalfa/module"
PROGRESS = "/lego/kalfa/progress"
FLOW_OUTPUTS = ["history", "predictions"]

MODEL_KEYS = ("inputs", "outputs", "nodes", "optimizer", "init", "ema", "trainable", "weights")
NODE_KEYS = ("uri", "block", "model", "params", "inputs", "outputs", "init", "repeat", "unpack")
LEGO_TYPES = ("model", "criterion", "objective", "metric", "schedule", "init", "pre", "preprocessor", "generate",
              "trigger")


def call(value):
    """A lego call as an inline component: a string is a call without params, a mapping keeps only uri and params."""
    if isinstance(value, str):
        return {"uri": value}
    out = {"uri": value["uri"]}
    params = value.get("params")
    if params:
        out["params"] = params
    return out


def call_with_params(value):
    """A lego call for a block variable read with ``$var.params$``: params is always present."""
    if isinstance(value, str):
        return {"uri": value, "params": {}}
    return {"uri": value["uri"], "params": dict(value.get("params") or {})}


def is_shortcut(section):
    return any(key in section for key in MODEL_KEYS)


def models_of(section):
    """(1) the single model shortcut becomes models.model; returns (templates, models)."""
    if is_shortcut(section):
        return {}, {"model": section}
    return dict(section.get("templates") or {}), dict(section.get("models") or {})


def is_composite(definition):
    """(3) a model whose graph uses a {model: name} node is a composite."""
    nodes = definition.get("nodes")
    items = nodes if isinstance(nodes, list) else list((nodes or {}).values())
    return any(isinstance(item, dict) and "model" in item for item in items)


def node_of(item):
    """A model node as a cirak block node: template → block, everything else kept in place."""
    out = {}
    for key in NODE_KEYS:
        if key == "block":
            if "template" in item:
                out["block"] = item["template"]
        elif key in item:
            out[key] = item[key]
    return out


def block_of(definition, template=False):
    """(10) a model or template definition as a cirak block: a list of nodes is a spec, a mapping a graph."""
    block = {}
    if template and definition.get("variables"):
        block["variables"] = definition["variables"]
    if "inputs" in definition:
        block["inputs"] = definition["inputs"]
    if "outputs" in definition:
        block["outputs"] = definition["outputs"]
    nodes = definition.get("nodes")
    inputs = definition.get("inputs") or []
    outputs = definition.get("outputs") or []
    if isinstance(nodes, list) and len(inputs) <= 1 and len(outputs) <= 1:
        block["spec"] = [node_of(item) for item in nodes]
    elif isinstance(nodes, list):
        block["graph"] = chain_graph(nodes, inputs, outputs)
    else:
        block["graph"] = {name: node_of(item) for name, item in (nodes or {}).items()}
    return block


def chain_graph(nodes, inputs, outputs):
    """A chain whose boundary is not one wire in and one wire out, as a graph: the first node takes every input,
    each node feeds the next, the last one writes the outputs."""
    graph = {}
    previous = list(inputs)
    for position, item in enumerate(nodes):
        node = node_of(item)
        node["inputs"] = previous
        last = position == len(nodes) - 1
        if last and len(outputs) == 1:
            name = outputs[0]
        elif last:
            name = f"s{position}"
            node["outputs"] = list(outputs)
        else:
            name = f"s{position}"
        graph[name] = node
        previous = [name]
    return graph


def blocks_of(templates, models):
    blocks = {name: block_of(definition, template=True) for name, definition in templates.items()}
    for name, definition in models.items():
        blocks[name] = block_of(definition)
    return blocks


def trained_and_composites(models):
    """(3)(4) trained models with their index in definition order, and the composites."""
    trained = []
    composites = []
    index = 0
    for name, definition in models.items():
        if is_composite(definition):
            composites.append({"name": name})
            continue
        trained.append({"name": name, "index": index, "init": definition.get("init"),
                        "trainable": bool(definition.get("trainable", True)), "weights": definition.get("weights")})
        index += 1
    return trained, composites


def ema_items(models):
    """(6) the models that write ema, with their decay."""
    found = []
    for name, definition in models.items():
        ema = definition.get("ema")
        if isinstance(ema, dict):
            found.append({"name": name, "decay": ema.get("decay")})
    return found


def optimizers_of(config, models):
    """(2)(5) the optimizer table with an inline optimizer under its model's name, and the models each one trains."""
    training = config.get("training") or {}
    table = {}
    for name, definition in (config.get("optimizers") or {}).items():
        table[name] = dict(definition)
    users = {}
    for name, definition in models.items():
        optimizer = definition.get("optimizer")
        if optimizer is None:
            continue
        if isinstance(optimizer, str):
            users.setdefault(optimizer, []).append(name)
        else:
            table[name] = {"uri": optimizer["uri"], "params": optimizer.get("params") or {},
                           "loss": training.get("loss"), "schedule": optimizer.get("schedule")}
            users.setdefault(name, []).append(name)
    items = []
    for name, definition in table.items():
        loss = definition.get("loss")
        if loss is None:
            loss = training.get("loss")
        items.append({"name": name, "uri": definition["uri"], "params": dict(definition.get("params") or {}),
                      "loss": loss, "schedule": definition.get("schedule"),
                      "models": {model: model for model in users.get(name, [])}})
    return items


def predicts_of(training, trained, composites):
    """predicts as written, else the only trained model, else None."""
    if training.get("predicts") is not None:
        return training["predicts"]
    if len(trained) == 1:
        return trained[0]["name"]
    return None


def lego_reference(value, aliases):
    """A string that names a lego, as the inline component cirak builds; a mapping is already one."""
    if isinstance(value, str):
        return {"uri": resolve_alias(value, aliases) or value}
    return value


def resolve_refs(uri, params, aliases, registry, preprocessors=None, generate=None):
    """The params of a lego call with its lego typed references (refs fact) turned into inline components.

    A ``preprocessor`` reference names a definition of data.preprocessors and becomes that definition's call; a
    ``generate`` reference written as the name ``generate`` becomes the generate section's call. References of type
    model, loss, field, column, wire, history and data stay names; the lego reads them at run time or the framework
    resolves them.
    """
    refs = registry.facts(uri).refs if isinstance(uri, str) else {}
    if not params or not refs:
        return params
    out = dict(params)
    for param, ref_type in refs.items():
        if param not in out:
            continue
        if ref_type in ("pre", "preprocessor") and isinstance(out[param], str):
            definition = (preprocessors or {}).get(out[param])
            if definition is not None:
                out[param] = call_resolved(definition, aliases, registry, preprocessors)
                continue
        if ref_type == "generate" and out[param] == "generate":
            if generate is None:
                raise ValueError(f"{uri}: {param} names the generate section, which the config does not write")
            out[param] = call_resolved(generate, aliases, registry, preprocessors)
            continue
        if ref_type in LEGO_TYPES and ref_type != "model":
            out[param] = lego_reference(out[param], aliases)
    return out


def call_resolved(value, aliases, registry, preprocessors=None, generate=None):
    """A lego call with its lego typed references resolved."""
    inner = call(value)
    if "params" in inner:
        inner["params"] = resolve_refs(inner["uri"], inner["params"], aliases, registry, preprocessors, generate)
    return inner


def set_values(targets, losses, aliases, registry):
    """Rule set values: a string for a param the lego declares as a lego reference becomes an inline component."""
    out = {}
    for key, value in (targets or {}).items():
        owner, _, param = key.partition(".")
        entry = losses.get(owner) if isinstance(losses, dict) else None
        if param and param != "loss" and isinstance(entry, dict) and isinstance(value, str):
            ref_type = registry.facts(entry.get("uri", "")).refs.get(param)
            if ref_type in LEGO_TYPES and ref_type != "model":
                out[key] = lego_reference(value, aliases)
                continue
        out[key] = value
    return out


def triggers_of(training, aliases=None, registry=None):
    registry = registry if registry is not None else default_registry
    aliases = aliases or {}
    found = {}
    for rule in training.get("rules") or []:
        found[rule["name"]] = call_resolved(rule["when"], aliases, registry)
    for position, trigger in enumerate(training.get("stop") or []):
        found[f"stop_{position}"] = call_resolved(trigger, aliases, registry)
    return found


def component_of(entry, registry, aliases=None, generate=None):
    """(11) a losses or metrics entry: criteria and metrics are wrapped in their adapter, objectives stay direct."""
    inner = call_resolved(entry, aliases or {}, registry, generate=generate)
    kind = kalfa_kind(inner["uri"])
    if kind == "criterion":
        return {"uri": ADAPTER_CRITERION, "params": {"criterion": inner}}
    if kind == "metric":
        return {"uri": ADAPTER_METRIC, "params": {"metric": inner}}
    return inner


def components_of(section, registry, aliases=None, generate=None):
    return {name: component_of(entry, registry, aliases, generate) for name, entry in (section or {}).items()}


def keys_of(section):
    """(12) the definition level keys of a component table, one entry per name (empty when nothing is written)."""
    table = {}
    for name, entry in (section or {}).items():
        table[name] = {key: entry[key] for key in DEFINITION_KEYS if isinstance(entry, dict) and key in entry}
    return table


def data_params(data, aliases=None, registry=None):
    """(7)(8)(9) the data block variables: split and batch short forms, string and mapping filters."""
    registry = registry if registry is not None else default_registry
    aliases = aliases or {}
    filters = data.get("filter") or []
    split = data["split"]
    if isinstance(split, dict) and "uri" in split:
        split = call_with_params(split)
    else:
        split = {"uri": RANDOM_SPLIT, "params": dict(split)}
    batch = data["batch"]
    batch = {"size": batch} if not isinstance(batch, dict) else dict(batch)
    table = data.get("preprocessors") or {}
    preprocessors = {name: call_resolved(entry, aliases, registry, table) for name, entry in table.items()}
    return {"source": call_with_params(data["source"]),
            "filter_pre": [item for item in filters if isinstance(item, str)],
            "filter_set": [item for item in filters if isinstance(item, dict)],
            "split": split,
            "batch": batch,
            "preprocessors": preprocessors,
            "preprocessors_keys": keys_of(data.get("preprocessors")),
            "drop": list(data.get("drop") or []),
            "fields": dict(data.get("fields") or {}),
            "feed": call_with_params(data["feed"])}


def recipe(config, registry=None, aliases=None):
    """The cirak document for a resolved config: component tables, model blocks and the five block usages."""
    registry = registry if registry is not None else default_registry
    aliases = aliases or {}
    training = config.get("training") or {}
    templates, models = models_of(config["model"])
    trained, composites = trained_and_composites(models)
    emas = ema_items(models)
    optimizers = optimizers_of(config, models)
    predicts = predicts_of(training, trained, composites)
    losses = config.get("losses") or {}
    rules = [{"name": rule["name"], "when": f"@triggers.{rule['name']}",
              "set": set_values(rule.get("set"), losses, aliases, registry), "after": rule.get("after")}
             for rule in training.get("rules") or []]
    checkpoint = training.get("checkpoint")
    generate = config.get("generate")
    document = {
        "losses": components_of(losses, registry, aliases, generate),
        "metrics": components_of(config.get("metrics"), registry, aliases, generate),
        "triggers": triggers_of(training, aliases, registry),
        "plots": {name: call_resolved(entry, aliases, registry) for name, entry in (config.get("plots") or {}).items()},
        "progress": {"uri": PROGRESS},
        "blocks": blocks_of(templates, models),
        "flow": {
            "outputs": list(FLOW_OUTPUTS),
            "data": {"block": "data", "params": data_params(config["data"], aliases, registry)},
            "models": {"block": "models", "params": {
                "trained_items": trained,
                "composite_items": composites,
                "ema_items": emas,
                "trained_refs": {item["name"]: item["name"] for item in trained},
                "composite_refs": {item["name"]: item["name"] for item in composites},
                "ema_refs": {item["name"]: f"{item['name']}_ema" for item in emas},
                "seed": config.get("seed")}},
            "optimizers": {"block": "optimizers", "params": {
                "optimizer_items": optimizers,
                "optimizer_refs": {item["name"]: f"opt_{item['name']}" for item in optimizers}}},
            "training": {"block": "training", "params": {
                "turn": call_with_params(training["turn"]),
                "turn_params": {key: value for key, value in training.items() if key not in TRAINING_FIXED},
                "losses_keys": keys_of(losses),
                "metrics_keys": keys_of(config.get("metrics")),
                "predicts": predicts,
                "epochs": training.get("epochs"),
                "steps": training.get("steps"),
                "rules": rules,
                "stop": [f"@triggers.stop_{position}" for position in range(len(training.get("stop") or []))],
                "checkpoint": None if checkpoint is None else call(checkpoint)}},
            "after": {"block": "after", "params": {
                "report": training.get("report"),
                "predicts": predicts,
                "generate": None if generate is None else call_resolved(generate, aliases, registry),
                "plots_keys": keys_of(config.get("plots"))}},
        },
    }
    return document
