from . import state, wrap
from .errors import BuildError, KalfaError


def as_list(value):
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return list(value)


def build_model(node, module, keys, ctx):
    name = node.name
    data_inputs = getattr(ctx.data, "inputs", None) if ctx.data is not None else None
    inputs = as_list(keys.get("inputs")) or list(data_inputs or ("x",))
    outputs = as_list(keys.get("outputs")) or [name]
    init = keys.get("init")
    if init is not None:
        if not callable(init):
            raise BuildError(f"{node.path}: init must be a call such as {{use: normal_init}}, got {init!r}")
        init(module)
    weights = keys.get("weights")
    if weights is not None:
        load_weights(node, module, weights)
    freeze = keys.get("freeze", False)
    entry = wrap.ModelEntry(name, module, inputs, outputs, freeze=freeze, weights=weights, init=init)
    if "freeze" in keys:
        entry.set_freeze(freeze)
    ema = keys.get("ema")
    if ema is not None and ema is not False:
        entry.ema = ema_copy(node, module, ema)
    entry.path = node.path
    return entry


def ema_copy(node, module, ema):
    if isinstance(ema, dict):
        decay = ema.get("decay", 0.999)
        every = ema.get("every", 1)
    elif ema is True:
        decay, every = 0.999, 1
    else:
        decay, every = ema, 1
    try:
        return wrap.EmaCopy(module, float(decay), int(every))
    except (TypeError, ValueError) as exc:
        raise BuildError(f"{node.path}: ema takes a decay such as 0.999 or {{decay, every}}, got {ema!r}") from exc


def load_weights(node, module, weights):
    spec = {"run": weights} if isinstance(weights, str) else weights
    if not isinstance(spec, dict) or not spec.get("run"):
        raise BuildError(f"{node.path}: weights takes {{run, model, which}}, got {weights!r}")
    model = spec.get("model") or node.name
    try:
        path = state.checkpoint_path(spec["run"], spec.get("which"))
        checkpoint = state.read_checkpoint(path)
    except KalfaError:
        raise
    except Exception as exc:
        raise BuildError(f"{node.path}: weights could not be read from {spec['run']}: {exc}") from exc
    nodes = checkpoint.get("nodes", {}) if isinstance(checkpoint, dict) else {}
    key = f"models.{model}"
    if key not in nodes:
        raise BuildError(f"{node.path}: {path} holds no {key}; it holds {sorted(nodes)}")
    try:
        module.load_state_dict(nodes[key])
    except Exception as exc:
        raise BuildError(f"{node.path}: the weights of {key} in {path} do not fit {node.use}: {exc}") from exc


def build_optimizer(node, kwargs, keys, signature, ctx):
    over = groups_of(node, keys.get("over"))
    entry = wrap.OptimizerEntry(node.name, node.target, kwargs, over, minimizes=keys.get("minimizes"),
                                schedule=keys.get("schedule"))
    entry.path = node.path
    link_emas(entry, over, ctx)
    try:
        if callable(getattr(entry, "bind", None)):
            entry.bind(ctx)
        else:
            entry.build()
    except KalfaError:
        raise
    except Exception as exc:
        raise BuildError(f"{node.path}: {node.use} could not be built: {type(exc).__name__}: {exc}") from exc
    return entry


def covers(model, module):
    return any(module is part for part in model.module.modules())


def link_emas(entry, over, ctx):
    emas = getattr(entry, "emas", None)
    if not isinstance(emas, list):
        return
    modules = [item.of if isinstance(item, wrap.Group) else item for item in over]
    for model in ctx.models.values():
        if model.ema is None or any(model.ema is known for known in emas):
            continue
        if any(covers(model, module) for module in modules):
            emas.append(model.ema)


def groups_of(node, over):
    if over is None:
        raise BuildError(f"{node.path}: over names the modules to optimize")
    items = over if isinstance(over, list) else [over]
    groups = []
    for item in items:
        if isinstance(item, dict):
            rest = dict(item)
            of = rest.pop("of", None)
            if of is None:
                raise BuildError(f"{node.path}: a group in over needs of: '@models.<name>'")
            groups.append(wrap.Group(of, rest.pop("match", None), **rest))
        else:
            groups.append(item)
    return groups
