from cirak.registry import registry

from .config import resolve_alias
from .contract import Contract
from .kinds import kalfa_kind
from .schema import Schema


def call(value):
    if isinstance(value, str):
        return {"uri": value}
    out = {"uri": value["uri"]}
    params = value.get("params")
    if params:
        out["params"] = params
    return out


def call_with_params(value):
    if isinstance(value, str):
        return {"uri": value, "params": {}}
    return {"uri": value["uri"], "params": dict(value.get("params") or {})}


def is_shortcut(section):
    return any(key in section for key in Schema.definition_of_model)


def models_of(section):
    if is_shortcut(section):
        return {}, {"model": section}
    return dict(section.get("templates") or {}), dict(section.get("models") or {})


def is_composite(definition):
    nodes = definition.get("nodes")
    items = nodes if isinstance(nodes, list) else list((nodes or {}).values())
    return any(isinstance(item, dict) and "model" in item for item in items)


def node_of(item):
    out = {}
    for key in Schema.node:
        if key not in item:
            continue
        out["block" if key == "template" else key] = item[key]
    return out


def block_of(definition, template=False):
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
    found = []
    for name, definition in models.items():
        ema = definition.get("ema")
        if isinstance(ema, dict):
            found.append({"name": name, "decay": ema.get("decay")})
    return found


def optimizers_of(config, models):
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
    if training.get("predicts") is not None:
        return training["predicts"]
    if len(trained) == 1:
        return trained[0]["name"]
    return None


def lego_reference(value, aliases):
    if isinstance(value, str):
        return {"uri": resolve_alias(value, aliases) or value}
    return value


def resolve_refs(uri, params, aliases, catalog, preprocessors=None, generate=None):
    refs = catalog.facts(uri).refs if isinstance(uri, str) else {}
    if not params or not refs:
        return params
    out = dict(params)
    for param, ref_type in refs.items():
        if param not in out:
            continue
        if ref_type in ("pre", "preprocessor") and isinstance(out[param], str):
            definition = (preprocessors or {}).get(out[param])
            if definition is not None:
                out[param] = call_resolved(definition, aliases, catalog, preprocessors)
                continue
        if ref_type == "generate" and out[param] == "generate":
            if generate is None:
                raise ValueError(f"{uri}: {param} names the generate section, which the config does not write")
            out[param] = call_resolved(generate, aliases, catalog, preprocessors)
            continue
        if Schema.ref(ref_type).lego:
            out[param] = lego_reference(out[param], aliases)
    return out


def call_resolved(value, aliases, catalog, preprocessors=None, generate=None):
    inner = call(value)
    if "params" in inner:
        inner["params"] = resolve_refs(inner["uri"], inner["params"], aliases, catalog, preprocessors, generate)
    return inner


def set_values(targets, losses, aliases, catalog):
    out = {}
    for key, value in (targets or {}).items():
        owner, _, param = key.partition(".")
        entry = losses.get(owner) if isinstance(losses, dict) else None
        if param and param != "loss" and isinstance(entry, dict) and isinstance(value, str):
            ref_type = catalog.facts(entry.get("uri", "")).refs.get(param)
            if Schema.ref(ref_type).lego:
                out[key] = lego_reference(value, aliases)
                continue
        out[key] = value
    return out


def triggers_of(training, aliases=None, catalog=None):
    catalog = catalog if catalog is not None else registry
    aliases = aliases or {}
    found = {}
    for rule in training.get("rules") or []:
        found[rule["name"]] = call_resolved(rule["when"], aliases, catalog)
    for position, trigger in enumerate(training.get("stop") or []):
        found[f"stop_{position}"] = call_resolved(trigger, aliases, catalog)
    return found


def component_of(entry, catalog, aliases=None, generate=None, contract=None):
    contract = contract or Contract.load()
    inner = call_resolved(entry, aliases or {}, catalog, generate=generate)
    kind = kalfa_kind(inner["uri"])
    adapter = contract.wiring.get(f"{kind}_adapter")
    if adapter is None:
        return inner
    return {"uri": adapter, "params": {kind: inner}}


def components_of(section, catalog, aliases=None, generate=None, contract=None):
    return {name: component_of(entry, catalog, aliases, generate, contract) for name, entry in (section or {}).items()}


def keys_of(section, targets=None):
    targets = dict(targets or {})
    table = {}
    for name, entry in (section or {}).items():
        keys = {key: entry[key] for key in Schema.definition if isinstance(entry, dict) and key in entry}
        if targets and keys.get("target") is None:
            inherited = targets.get(keys["output"]) if keys.get("output") is not None else (
                next(iter(targets.values())) if len(targets) == 1 else None)
            if inherited is not None:
                keys["target"] = inherited
        table[name] = keys
    return table


def plots_keys_of(plots):
    table = keys_of(plots)
    for name, entry in (plots or {}).items():
        table[name]["lego"] = call(entry)["uri"]
    return table


def rng_of(config, contract):
    if config.get("rng") is not None:
        return call_with_params(config["rng"])
    return {"uri": contract.wiring["default_rng"], "params": {}}


def loaders_of(batch, contract):
    batch = {"size": batch} if not isinstance(batch, dict) else dict(batch)
    return {name: {"uri": contract.wiring["loader"], "set": name, "params": {"set": name, **batch}}
            for name in contract.sets}


def prep_of(data, preprocessors, keys, contract, record=None):
    if record is not None:
        return {"uri": contract.wiring["read_prep"], "params": {"record": str(record)}, "inputs": {}}
    params = {"fields": dict(data.get("fields") or {}), "preprocessors": preprocessors,
              "drop": list(data.get("drop") or []), "keys": keys}
    return {"uri": contract.wiring["fit"], "params": params, "inputs": {"df": "train_df"}}


def data_params(data, aliases=None, catalog=None, contract=None, record=None):
    catalog = catalog if catalog is not None else registry
    contract = contract or Contract.load()
    aliases = aliases or {}
    filters = data.get("filter") or []
    split = data["split"]
    if isinstance(split, dict) and "uri" in split:
        split = call_with_params(split)
    else:
        split = {"uri": contract.wiring["default_split"], "params": dict(split)}
    table = data.get("preprocessors") or {}
    preprocessors = {name: call_resolved(entry, aliases, catalog, table) for name, entry in table.items()}
    keys = keys_of(data.get("preprocessors"))
    return {"source": call_with_params(data["source"]),
            "filter_pre": [item for item in filters if isinstance(item, str)],
            "filter_set": [item for item in filters if isinstance(item, dict)],
            "split": split,
            "loaders": loaders_of(data["batch"], contract),
            "prep": prep_of(data, preprocessors, keys, contract, record),
            "preprocessors_keys": keys,
            "feed": call_with_params(data["feed"])}


def recipe(config, catalog=None, aliases=None, contract=None, record=None):
    catalog = catalog if catalog is not None else registry
    contract = contract or Contract.load()
    aliases = aliases or {}
    training = config.get("training") or {}
    templates, models = models_of(config["model"])
    trained, composites = trained_and_composites(models)
    emas = ema_items(models)
    optimizers = optimizers_of(config, models)
    predicts = predicts_of(training, trained, composites)
    losses = config.get("losses") or {}
    rules = [{"name": rule["name"], "when": f"@triggers.{rule['name']}",
              "set": set_values(rule.get("set"), losses, aliases, catalog), "after": rule.get("after")}
             for rule in training.get("rules") or []]
    checkpoint = training.get("checkpoint")
    targets = training.get("targets") or {}
    generate = config.get("generate")
    document = {
        "losses": components_of(losses, catalog, aliases, generate, contract),
        "metrics": components_of(config.get("metrics"), catalog, aliases, generate, contract),
        "triggers": triggers_of(training, aliases, catalog),
        "plots": {name: call_resolved(entry, aliases, catalog) for name, entry in (config.get("plots") or {}).items()},
        "checkpoint": call(checkpoint) if checkpoint is not None else {},
        "blocks": blocks_of(templates, models),
        "flow": {
            "outputs": ["history", "predictions"],
            "data": {"block": "data", "params": data_params(config["data"], aliases, catalog, contract, record)},
            "models": {"block": "models", "params": {
                "trained_items": trained,
                "composite_items": composites,
                "ema_items": emas,
                "trained_refs": {item["name"]: item["name"] for item in trained},
                "composite_refs": {item["name"]: item["name"] for item in composites},
                "ema_refs": {item["name"]: f"{item['name']}_ema" for item in emas},
                "seed": config.get("seed"),
                "rng": rng_of(config, contract),
                "builder": contract.wiring["builder"]}},
            "optimizers": {"block": "optimizers", "params": {
                "optimizer_items": optimizers,
                "optimizer_refs": {item["name"]: f"opt_{item['name']}" for item in optimizers}}},
            "training": {"block": "training", "params": {
                "turn": call_with_params(training["turn"]),
                "turn_params": {key: value for key, value in training.items() if key not in Schema.training_fixed},
                "losses_keys": keys_of(losses, targets),
                "metrics_keys": keys_of(config.get("metrics"), targets),
                "predicts": predicts,
                "epochs": training.get("epochs"),
                "steps": training.get("steps"),
                "rules": rules,
                "stop": [f"@triggers.stop_{position}" for position in range(len(training.get("stop") or []))],
                "history_prefix": contract.history_prefix}},
            "after": {"block": "after", "params": {
                "report": training.get("report"),
                "figures": {"uri": contract.wiring["figures"], "params": dict(config.get("figures") or {})},
                "predicts": predicts,
                "targets": targets,
                "generate": None if generate is None else call_resolved(generate, aliases, catalog),
                "plots_keys": plots_keys_of(config.get("plots")),
                "plot_bus": contract.plot_bus,
                "losses_keys": keys_of(losses, targets)}},
        },
    }
    return document
