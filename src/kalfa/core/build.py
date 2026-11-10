import functools
import inspect
from dataclasses import dataclass, field

from . import aliases, wrap
from .coerce import coerced
from .context import train_of
from .errors import BuildError, KalfaError, ProblemsError, error
from .make import build_model, build_optimizer
from .rules import parse_rules


SLOT_KEYS = {
    "data": (),
    "model": ("inputs", "outputs", "ema", "freeze", "weights", "init"),
    "loss": ("pred", "target", "on", "every", "rescale"),
    "metric": ("pred", "target", "on", "every", "rescale"),
    "optimizer": ("over", "minimizes", "schedule"),
    "step": ("order", "amp", "grad_clip", "accumulate"),
    "trigger": (),
    "action": (),
    "logger": (),
    "schedule": (),
    "plot": (),
    "sampler": (),
    "strategy": (),
    "value": (),
}

SECTION_SLOTS = {
    "data": "data",
    "models": "model",
    "losses": "loss",
    "metrics": "metric",
    "optimizers": "optimizer",
    "plots": "plot",
    "generate": "sampler",
}

ORDER = ("data", "models", "losses", "metrics", "optimizers", "train.step", "plots", "generate")

MODEL_FIXED = ("inputs", "outputs", "ema", "weights", "init")
OPTIMIZER_FIXED = ("over", "schedule")
CALL_KEYS = {"model": ("init",), "optimizer": ("schedule",)}

TOP_LEVEL = ("include", "params", "plugins", "seed", "device", "record", "data", "models", "losses", "metrics",
             "optimizers", "train", "rules", "plots", "generate", "sweep")
LOADER_KEYS = ("include", "params", "plugins")
SINGLE_NODES = ("data", "generate")
DEFAULT_STEP = "supervised"
STEP_ATTRIBUTES = SLOT_KEYS["step"]


@dataclass
class Ref:
    text: str
    path: str
    attr: str | None
    at: str
    target: object = None

    @property
    def name(self):
        return self.path.rsplit(".", 1)[-1]


@dataclass
class Node:
    path: str
    slot: str
    use: str | None
    target: object
    params: dict
    keys: dict = field(default_factory=dict)
    refs: list = field(default_factory=list)
    children: list = field(default_factory=list)
    obj: object = None

    @property
    def name(self):
        return self.path.rsplit(".", 1)[-1]

    @property
    def section(self):
        return self.path.split(".", 1)[0]


@dataclass
class Signature:
    params: dict
    accepts_extra: bool
    required: tuple


@dataclass
class Tree:
    config: object
    nodes: dict
    order: list
    settings: dict
    rules: list
    problems: list = field(default_factory=list)
    imports: list = field(default_factory=list)

    @property
    def errors(self):
        return [problem for problem in self.problems if problem.severity == "error"]

    def node(self, path):
        return self.nodes[path]

    def section(self, name):
        return {node.name: node for node in self.nodes.values()
                if node.path.startswith(f"{name}.") and "." not in node.path[len(name) + 1:]
                and "[" not in node.path[len(name) + 1:]}


def signature_of(target):
    if inspect.isclass(target):
        function = target.__init__
        skip = 1
    else:
        function = target
        skip = 0
    try:
        raw = inspect.signature(function)
    except (TypeError, ValueError):
        return Signature({}, True, ())
    params = {}
    extra = False
    required = []
    for name, parameter in list(raw.parameters.items())[skip:]:
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            extra = True
            continue
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        params[name] = parameter
        if parameter.default is inspect.Parameter.empty:
            required.append(name)
    return Signature(params, extra, tuple(required))


def leading_required(signature):
    count = 0
    positional = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    for parameter in signature.params.values():
        if parameter.kind not in positional or parameter.default is not inspect.Parameter.empty:
            break
        count += 1
    return count


def criterion_function(target, signature=None):
    if target is None or inspect.isclass(target) or not callable(target):
        return False
    signature = signature_of(target) if signature is None else signature
    return leading_required(signature) >= 2


def node_signature(target, slot):
    signature = signature_of(target)
    drop = ()
    names = list(signature.params)
    if slot == "optimizer" and names:
        drop = ("params",) if "params" in names else (names[0],)
    elif slot == "plot" and not inspect.isclass(target):
        drop = tuple(names[:2])
    elif slot == "loss" and criterion_function(target, signature):
        drop = tuple(names[:2])
    if not drop:
        return signature
    params = {name: parameter for name, parameter in signature.params.items() if name not in drop}
    required = tuple(name for name in signature.required if name not in drop)
    return Signature(params, signature.accepts_extra, required)


def fixed_of(node):
    target = node.target
    if node.slot == "model":
        names = list(MODEL_FIXED)
        if target is not None:
            names.extend(node_signature(target, node.slot).params)
        names.extend(name for name in node.params if name not in names)
        return tuple(dict.fromkeys(names))
    if node.slot in ("loss", "metric"):
        return tuple(dict.fromkeys(("pred", "target", *declared_fixed(target))))
    if node.slot == "optimizer":
        return OPTIMIZER_FIXED
    if node.slot in ("step", "trigger"):
        return tuple(declared_fixed(target))
    return ()


def declared_fixed(target):
    found = getattr(target, "fixed", ()) if target is not None else ()
    if isinstance(found, str):
        return (found,)
    try:
        return tuple(str(name) for name in found)
    except TypeError:
        return ()


def join(path, key):
    return f"{path}[{key}]" if isinstance(key, int) else f"{path}.{key}"


class Builder:
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.problems = []
        self.declared = []
        self.known = set()
        self.nodes = {}
        self.settings = {}
        self.rules = {}
        self.imports = []

    def fail(self, code, message, path=None, hint=None):
        self.problems.append(error(code, message, path, hint))

    def run(self):
        self.scan_top_level()
        self.declare_all()
        self.known = {path for path, _, _ in self.declared}
        for path, slot, raw in self.declared:
            self.add(self.make_node(path, slot, raw))
        specs = parse_rules(self.rules or None, self.problems)
        return Tree(self.config, self.nodes, list(self.nodes), self.settings, specs, self.problems, self.imports)

    def scan_top_level(self):
        for key, value in self.data.items():
            name = str(key)
            if name not in TOP_LEVEL:
                self.fail("unknown_section", f"{name!r} is not a top level key; the keys are {list(TOP_LEVEL)}", name)
                self.settings[name] = value
                continue
            if name in LOADER_KEYS or name in SECTION_SLOTS:
                continue
            if name == "rules":
                if value is None:
                    continue
                if not isinstance(value, dict):
                    self.fail("bad_section", "rules must be a mapping of rule names to rules", "rules")
                    continue
                self.rules = value
            elif name == "train":
                train = self.train_mapping()
                self.settings["train"] = {k: v for k, v in train.items() if k != "step"}
            else:
                self.settings[name] = value

    def train_mapping(self):
        train = self.data.get("train")
        if train is None:
            return {}
        if not isinstance(train, dict):
            self.fail("bad_section", "train must be a mapping with epochs or updates", "train")
            return {}
        return train

    def declare(self, path, slot, raw):
        self.declared.append((path, slot, raw))

    def declare_all(self):
        for section in ORDER:
            if section == "train.step":
                train = self.data.get("train") if isinstance(self.data.get("train"), dict) else {}
                step = train.get("step")
                if step is None:
                    step = {"use": DEFAULT_STEP}
                self.declare("train.step", "step", step)
                log = train.get("log")
                for index, entry in enumerate([log] if isinstance(log, (str, dict)) else list(log or [])):
                    self.declare(f"train.log[{index}]", "logger", entry)
            elif section in SINGLE_NODES:
                if section in self.data and self.data[section] is not None:
                    self.declare(section, SECTION_SLOTS[section], self.data[section])
            else:
                self.declare_section(section)
        rules = self.rules if isinstance(self.rules, dict) else {}
        for name, rule in rules.items():
            if not isinstance(rule, dict):
                continue
            if rule.get("when") is not None:
                self.declare(f"rules.{name}.when", "trigger", rule["when"])
            if rule.get("run") is not None:
                self.declare(f"rules.{name}.run", "action", rule["run"])

    def declare_section(self, section):
        raw = self.data.get(section)
        if raw is None:
            return
        if not isinstance(raw, dict):
            self.fail("bad_section", f"{section} must be a mapping of names to calls", section)
            return
        for name, value in raw.items():
            self.declare(f"{section}.{name}", SECTION_SLOTS[section], value)

    def add(self, node):
        self.nodes[node.path] = node
        for child in node.children:
            self.add(child)

    def make_node(self, path, slot, raw):
        if isinstance(raw, str) and not raw.startswith("@"):
            raw = {"use": raw}
        node = Node(path, slot, None, None, {})
        if not isinstance(raw, dict) or "use" not in raw:
            self.fail("not_a_call", f"expected a call such as {{use: name, ...}} or a name, got {raw!r}", path,
                      "a mapping is a call only when it carries use")
            return node
        text = raw["use"]
        if not isinstance(text, str) or not text:
            self.fail("not_a_call", f"use must be a short name or an import path, got {text!r}", path)
            return node
        node.use = text
        resolved = None
        try:
            resolved = aliases.resolve(text)
        except BuildError as exc:
            self.fail("unknown_use", str(exc), path, aliases.suggestion(text))
        if resolved is not None:
            node.use = aliases.target_path(resolved)
            try:
                node.target = aliases.import_target(resolved)
            except BuildError as exc:
                self.fail("import_failed", str(exc), path)
            if node.target is not None and aliases.is_import_path(node.use) and node.use not in self.imports:
                self.imports.append(node.use)
        slot_keys = SLOT_KEYS.get(slot, ())
        for key, value in raw.items():
            if key == "use":
                continue
            if not isinstance(key, str):
                self.fail("bad_key", f"the key {key!r} is not a name; quote keys such as yes, no, on and off", path)
                continue
            if key in CALL_KEYS.get(slot, ()) and isinstance(value, str) and not value.startswith("@"):
                value = {"use": value}
            holder = node.keys if key in slot_keys else node.params
            holder[key] = self.value(value, join(path, key), node)
        if slot == "optimizer":
            self.check_optimizer(node)
        if node.target is not None:
            self.check_signature(node)
        return node

    def value(self, raw, at, node):
        if isinstance(raw, dict):
            if "use" in raw:
                child = self.make_node(at, "value", raw)
                node.children.append(child)
                return child
            return {key: self.value(item, join(at, key), node) for key, item in raw.items()}
        if isinstance(raw, (list, tuple)):
            return [self.value(item, join(at, index), node) for index, item in enumerate(raw)]
        if isinstance(raw, str) and raw.startswith("@"):
            ref = self.reference(raw, at, node)
            node.refs.append(ref)
            return ref
        return raw

    def reference(self, text, at, node):
        body = text[1:]
        parts = [part for part in body.split(".") if part]
        if parts and len(parts) == body.count(".") + 1:
            for cut in range(len(parts), 0, -1):
                path = ".".join(parts[:cut])
                if path in self.known:
                    attr = ".".join(parts[cut:]) or None
                    return Ref(text, path, attr, at)
        root = parts[0] if parts else ""
        if root in SECTION_SLOTS or root == "train":
            known = sorted(path for path in self.known if path == root or path.startswith(f"{root}."))
            message = f"{at}: {text} names nothing; {root} has {known}"
        else:
            roots = ", ".join([*SECTION_SLOTS, "train.step"])
            message = f"{at}: {text} names nothing; a reference starts with one of {roots}"
        self.fail("reference_unknown", message, node.path)
        return Ref(text, body, None, at)

    def check_optimizer(self, node):
        over = node.keys.get("over")
        if "over" not in node.keys:
            self.fail("missing_key", "an optimizer needs over, the modules it optimizes, such as over: '@models.net'",
                      node.path)
        else:
            items = over if isinstance(over, list) else [over]
            for index, item in enumerate(items):
                where = join(join(node.path, "over"), index) if isinstance(over, list) else join(node.path, "over")
                if isinstance(item, dict):
                    if not isinstance(item.get("of"), Ref):
                        self.fail("reference_expected", f"{where}: a group needs of: '@models.<name>'", node.path)
                elif not isinstance(item, Ref):
                    self.fail("reference_expected", f"{where}: expected '@models.<name>', got {item!r}", node.path)
        minimizes = node.keys.get("minimizes")
        if minimizes is not None and not isinstance(minimizes, Ref):
            self.fail("reference_expected", f"{join(node.path, 'minimizes')}: expected '@losses.<name>', got "
                                            f"{minimizes!r}", node.path)

    def check_signature(self, node):
        signature = node_signature(node.target, node.slot)
        known = signature.params
        label = node.use
        if not signature.accepts_extra:
            for key in node.params:
                if key not in known:
                    self.fail("unknown_key", f"{label} has no param {key!r}; its params are {sorted(known)}",
                              node.path)
        provided = set(node.params)
        if node.slot == "step":
            provided.update(node.keys)
        missing = [name for name in signature.required if name not in provided]
        if missing:
            self.fail("missing_key", f"{label} needs {missing}", node.path)
        if node.slot != "step":
            for key in node.keys:
                if key in known:
                    self.fail("slot_conflict", f"{key!r} is a {node.slot} slot key and also a param of {label}; "
                                               f"the class cannot take it from the config", node.path,
                              "rename the constructor param")


def build(config):
    data = config if isinstance(config, dict) else getattr(config, "data", None)
    if not isinstance(data, dict):
        data = {}
    return Builder(config, data).run()


def materialize(value):
    if isinstance(value, Node):
        return value.obj
    if isinstance(value, Ref):
        return value.target
    if isinstance(value, dict):
        return {key: materialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [materialize(item) for item in value]
    return value


def instantiate(tree, ctx):
    if tree.errors:
        raise ProblemsError(tree.errors)
    built = set()
    for path in tree.order:
        if path not in built:
            build_node(tree, ctx, tree.nodes[path], built)
    for section in ("models", "losses", "metrics", "optimizers"):
        table = getattr(ctx, section, None)
        if isinstance(table, dict):
            for name in tree.section(section):
                if name in table:
                    table[name] = table.pop(name)
    return ctx


def build_node(tree, ctx, node, built=None):
    built = set() if built is None else built
    built.add(node.path)
    for child in node.children:
        if child.path not in built:
            build_node(tree, ctx, child, built)
    resolve_refs(tree, ctx, [node.path], built)
    if node.target is None:
        raise BuildError(f"{node.path}: nothing to build, {node.use!r} did not import")
    kwargs = materialize(node.params)
    keys = materialize(node.keys)
    signature = node_signature(node.target, node.slot)
    slot = node.slot
    if slot == "optimizer":
        obj = build_optimizer(node, kwargs, keys, signature, ctx)
        ctx.optimizers[node.name] = obj
    elif slot == "plot":
        if inspect.isclass(node.target):
            obj = construct(node, kwargs)
            label(obj, node, signature)
            bind(obj, ctx)
            ctx.plots[node.name] = (obj, {})
        else:
            obj = node.target
            ctx.plots[node.name] = (obj, kwargs)
    else:
        if slot == "step":
            kwargs = {**kwargs, **{key: value for key, value in keys.items() if key in signature.params}}
        if slot == "loss" and criterion_function(node.target):
            obj = functools.partial(node.target, **kwargs) if kwargs else node.target
        else:
            obj = construct(node, kwargs)
            label(obj, node, signature)
        if slot == "data":
            ctx.data = obj
            bind(obj, ctx)
            prepare_data(obj, ctx, node)
        elif slot == "model":
            bind(obj, ctx)
            ctx.models[node.name] = build_model(node, obj, keys, ctx)
        elif slot == "loss":
            bind(obj, ctx)
            wrapped = wrap.wrap_loss(obj, keys, node.name, node.path)
            if wrapped is not obj:
                bind(wrapped, ctx)
            ctx.losses[node.name] = wrapped
            obj = wrapped
        elif slot == "metric":
            bind(obj, ctx)
            wrapped = wrap.wrap_metric(obj, keys, node.name, node.path)
            if wrapped is not obj:
                bind(wrapped, ctx)
            ctx.metrics[node.name] = wrapped
            obj = wrapped
        elif slot == "step":
            for key in STEP_ATTRIBUTES:
                if key in keys:
                    setattr(obj, key, keys[key])
            bind(obj, ctx)
            ctx.step = obj
        elif slot == "sampler":
            bind(obj, ctx)
            ctx.generate = obj
        elif slot == "logger":
            if ctx.fitted_from is None:
                bind(obj, ctx)
                ctx.loggers.append(obj)
        else:
            bind(obj, ctx)
    node.obj = obj
    return obj


def construct(node, kwargs):
    try:
        obj = node.target(**coerced(node.target, kwargs))
    except ProblemsError:
        raise
    except KalfaError as exc:
        message = str(exc)
        if message.startswith((f"{node.path}:", f"{node.path}.", f"{node.path}[")):
            raise
        raise type(exc)(f"{node.path}: {message}") from exc
    except Exception as exc:
        raise BuildError(f"{node.path}: {node.use} could not be built: {type(exc).__name__}: {exc}") from exc
    if obj is None:
        raise BuildError(f"{node.path}: {node.use} returned None; a factory returns the object to use")
    return obj


def label(obj, node, signature):
    for attribute, value in (("name", node.name), ("path", node.path)):
        if attribute in signature.params:
            continue
        try:
            setattr(obj, attribute, value)
        except (AttributeError, TypeError):
            pass


def bind(obj, ctx):
    method = getattr(obj, "bind", None)
    if callable(method):
        method(ctx)


def prepare_data(obj, ctx, node):
    source = getattr(ctx, "fitted_from", None)
    if source is None:
        fit(obj, ctx, node)
        return
    if callable(getattr(obj, "load", None)):
        obj.load(str(source))
    if getattr(ctx, "want_sets", True) and callable(getattr(obj, "rebuild", None)):
        try:
            obj.rebuild(ctx)
        except KalfaError as exc:
            ctx.rebuild_failure = str(exc)
            ctx.logger.warning(f"the sets of {source} could not be rebuilt ({exc}); only the fitted preprocessors "
                               f"are loaded")


def fit(obj, ctx, node):
    method = getattr(obj, "fit", None)
    if not callable(method):
        return
    try:
        method(ctx)
    except KalfaError:
        raise
    except Exception as exc:
        raise BuildError(f"{node.path}: {node.use} could not fit: {type(exc).__name__}: {exc}") from exc


def resolve_refs(tree, ctx, paths=None, built=None):
    nodes = [tree.nodes[path] for path in paths] if paths is not None else list(tree.nodes.values())
    resolved = {}
    for node in nodes:
        for ref in node.refs:
            ref.target = locate(tree, ctx, ref, built)
            resolved[ref.at] = ref.target
    return resolved


def demand(tree, ctx, ref, built):
    node = tree.nodes.get(ref.path)
    if node is None or node.obj is not None or built is None:
        return
    if node.section != ref.at.split(".", 1)[0].split("[", 1)[0]:
        return
    if node.path in built:
        raise BuildError(f"{ref.at}: {ref.text} is being built and refers back to itself; the references form a "
                         f"cycle")
    build_node(tree, ctx, node, built)


def exposed(ref, base):
    first = ref.attr.split(".")[0]
    exposes = getattr(base, "exposes", None)
    if exposes is None:
        if first.startswith("_") or not hasattr(base, first):
            raise BuildError(f"{ref.at}: {ref.text}: the data class has no attribute {first!r}")
        return
    if first not in tuple(exposes):
        raise BuildError(f"{ref.at}: {ref.text} is not exposed; the data class exposes {list(exposes)}")


def locate(tree, ctx, ref, built=None):
    root, _, rest = ref.path.partition(".")
    if ref.path == "data":
        base = ctx.data
        if base is None:
            raise BuildError(f"{ref.at}: {ref.text} is asked before data is built")
        if ref.attr is None:
            return base
        exposed(ref, base)
        return walk(tree, ref, base, ref.attr)
    if root == "models" and rest:
        entry = ctx.models.get(rest)
        if entry is None:
            demand(tree, ctx, ref, built)
            entry = ctx.models.get(rest)
        if entry is None:
            raise BuildError(f"{ref.at}: {ref.text} is not built yet")
        if ref.attr is None:
            return entry.module
        parts = ref.attr.split(".")
        if parts[0] == "ema":
            if entry.ema is None:
                raise BuildError(f"{ref.at}: {ref.text} asks for an EMA copy but models.{rest} has no ema key")
            return walk(tree, ref, entry.ema, ".".join(parts[1:])) if len(parts) > 1 else entry.ema
        return walk(tree, ref, entry.module, ref.attr)
    tables = {"losses": ctx.losses, "metrics": ctx.metrics, "optimizers": ctx.optimizers}
    base = None
    if root in tables and rest:
        base = tables[root].get(rest)
    elif ref.path == "generate":
        base = ctx.generate
    elif ref.path == "train.step":
        base = ctx.step
    if base is None:
        demand(tree, ctx, ref, built)
        node = tree.nodes.get(ref.path)
        base = node.obj if node is not None else None
    if base is None:
        raise BuildError(f"{ref.at}: {ref.text} is not built yet")
    return walk(tree, ref, base, ref.attr) if ref.attr else base


def walk(tree, ref, base, attr):
    current = base
    done = []
    for part in attr.split("."):
        done.append(part)
        try:
            current = getattr(current, part)
        except AttributeError:
            child = tree.nodes.get(f"{ref.path}.{'.'.join(done)}")
            if child is not None and child.obj is not None:
                current = child.obj
            else:
                raise BuildError(f"{ref.at}: {ref.text}: {type(current).__name__} has no attribute {part!r}") from None
    return current


def param_names(node):
    names = list(node.params)
    if node.target is not None:
        signature = node_signature(node.target, node.slot)
        names.extend(name for name in signature.params if name not in names)
    return names


def leaves(path, value):
    yield path
    if isinstance(value, dict):
        for key, item in value.items():
            yield from leaves(join(path, key), item)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from leaves(join(path, index), item)


def mutable_paths(tree):
    found = set()
    if train_of(tree).get("updates"):
        found.add("train.updates.total")
    else:
        found.add("train.epochs")
    for node in tree.nodes.values():
        fixed = fixed_of(node)
        if node.slot == "optimizer":
            found.update(f"{node.path}.{name}" for name in param_names(node))
            found.add(f"{node.path}.minimizes")
        elif node.slot == "model":
            found.add(f"{node.path}.freeze")
        elif node.slot in ("loss", "metric", "step", "trigger"):
            names = param_names(node)
            if node.slot == "step":
                names.extend(name for name in STEP_ATTRIBUTES if name not in names)
            for name in names:
                if name in fixed:
                    continue
                found.update(leaves(join(node.path, name), node.params.get(name)))
            if node.slot in ("loss", "metric"):
                found.update((f"{node.path}.on", f"{node.path}.every"))
    return sorted(found)
