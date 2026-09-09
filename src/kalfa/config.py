"""Loading the user's config: layers and --set through cirak's loader, then kalfa's alias and $param$ resolution.

The loaded config keeps the user's shape; only lego names become full URIs and ``$param$`` placeholders
become values. ``$datetime$`` stays literal here and is filled when a record directory is opened.
"""

import sys
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path

from cirak.errors import Problem, error
from cirak.loader import Layer, load, parse_value, set_layer
from cirak.merge import describe_layers, merge_layers
from cirak.registry import registry
from cirak.resolve import TOKEN

from .std.log import logger_for

logger = logger_for("config")

BUILTIN_VARIABLES = ("datetime",)
SKIPPED_SECTIONS = ("params", "alias", "plugins", "include")
REFERENCE_TYPES = ("criterion", "objective", "metric", "schedule", "init", "generate", "trigger")
INIT_ROLES = ("weights", "bias", "scale")
SHORT_CALL_PATHS = (("training", "turn"), ("training", "checkpoint"), ("data", "feed"), ("sweep", "strategy"),
                    ("device",))


@dataclass
class Surface:
    """The user's config as loaded: raw (merged) and resolved trees, provenance, overrides and aliases."""

    data: dict
    raw: dict
    provenance: dict
    overrides: list
    layer: Layer
    aliases: dict
    problems: list = field(default_factory=list)
    paths: list = field(default_factory=list)

    @property
    def errors(self):
        return [problem for problem in self.problems if problem.severity == "error"]

    def source(self, path):
        return self.provenance.get(tuple(path))

    def layers_text(self):
        return describe_layers(self.layer, self.overrides)


TOP_KEYS = ("plugins", "params", "seed", "device", "data", "model", "metrics", "losses", "optimizers", "training",
            "generate", "plots", "record")


def parse_set(text):
    """``--set path=value``: a dotted path from the document root; a single segment must be a top level key."""
    path, separator, value = text.partition("=")
    if not separator or not path:
        raise ValueError(f"--set expects PATH=VALUE, got {text!r}")
    if "." not in path and path not in TOP_KEYS:
        raise ValueError(f"--set {text!r}: {path!r} is not a top level key; write a dotted path from the root "
                         f"(--set training.epochs=5) or -p {text} for params.{path}")
    return path, parse_value(value)


def parse_param(text):
    """``-p name=value``: the params entry ``name``."""
    name, separator, value = text.partition("=")
    if not separator or not name:
        raise ValueError(f"-p expects NAME=VALUE, got {text!r}")
    return f"params.{name}", parse_value(value)


def parse_sets(sets=None, params=None):
    """The top layer of a run from ``--set`` and ``-p`` texts, in the order given."""
    return [parse_set(text) for text in sets or []] + [parse_param(text) for text in params or []]


def _import_plugins(names, problems):
    for name in names or []:
        if not isinstance(name, str):
            problems.append(error("plugin_import_failed", f"plugin entries must be strings, got {name!r}"))
        elif name not in sys.modules:
            try:
                import_module(name)
                logger.info(f"plugin {name}")
            except Exception as exc:
                problems.append(error("plugin_import_failed", f"cannot import plugin {name}: {exc}"))


def _add_to_sys_path(directory):
    if directory.is_dir() and str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


def _extend_sys_path(layer):
    """Plugins import from the directory of every config file and from a plugins/ folder next to it (a record
    directory keeps the copies of the modules its run imported)."""
    for loaded in layer.walk():
        if loaded.file == "--set":
            continue
        parent = Path(loaded.file).parent
        for directory in (parent / "plugins", parent):
            _add_to_sys_path(directory)


def _module_name(text):
    path = Path(text)
    if path.suffix == ".py":
        _add_to_sys_path(path.resolve().parent)
        return path.stem
    for directory in (Path.cwd() / "plugins", Path.cwd()):
        _add_to_sys_path(directory)
    return text


def import_plugins(paths=(), modules=()):
    problems = []
    for path in paths or ():
        layer, load_problems = load([str(path)], registry.fragments())
        raw, _, _, merge_problems = merge_layers(layer)
        problems.extend([*load_problems, *merge_problems])
        _extend_sys_path(layer)
        _import_plugins(raw.get("plugins"), problems)
    _import_plugins([_module_name(name) for name in modules or ()], problems)
    return problems


def plugin_aliases():
    """Short names declared by legos outside kalfa's std, usable without a pack."""
    from .std import STD_URIS

    return {name: uri for name, uri in registry.aliases().items() if uri not in STD_URIS}


def load_surface(paths, sets=None) -> Surface:
    paths = [str(path) for path in paths]
    logger.info(f"config {', '.join(paths)}")
    layer, problems = load(paths, registry.fragments())
    if sets:
        layer = Layer(files=[], below=[layer, set_layer(sets)], label="")
    raw, provenance, overrides, merge_problems = merge_layers(layer)
    if overrides:
        logger.debug(f"{len(overrides)} overridden leaves")
    problems = [*problems, *merge_problems]
    _extend_sys_path(layer)
    _import_plugins(raw.get("plugins"), problems)
    aliases = dict(plugin_aliases())
    table = raw.get("alias")
    if isinstance(table, dict):
        aliases.update({name: target for name, target in table.items() if isinstance(target, str)})
    data = resolve_surface(raw, provenance, aliases, problems)
    return Surface(data, raw, provenance, overrides, layer, aliases, problems, paths)


def resolve_alias(text, aliases):
    """Follow a short name to its URI; a text starting with / is already one. None when unknown."""
    seen = []
    current = text
    while not current.startswith("/"):
        if current in seen or current not in aliases:
            return None
        seen.append(current)
        current = aliases[current]
    return current


def resolve_surface(raw, provenance, aliases, problems):
    globals_ = raw.get("params") if isinstance(raw.get("params"), dict) else {}
    resolver = _Resolver(aliases, globals_, provenance, problems)
    out = {}
    for key, value in raw.items():
        if key in SKIPPED_SECTIONS:
            out[key] = value
        else:
            if (key,) in SHORT_CALL_PATHS and isinstance(value, str):
                out[key] = resolver.uri(value, (key,))
            else:
                out[key] = resolver.walk(value, (key,))
    resolve_rule_sets(out, aliases)
    return out


def resolve_rule_sets(data, aliases):
    """Rule set values that name a lego by alias (the target loss param has a refs type of a lego kind) are written
    as URIs, like the params themselves, so that resolved.yaml reloads without the alias packs."""
    losses = data.get("losses") if isinstance(data.get("losses"), dict) else {}
    training = data.get("training") if isinstance(data.get("training"), dict) else {}
    for rule in training.get("rules") or []:
        if not isinstance(rule, dict) or not isinstance(rule.get("set"), dict):
            continue
        for key, value in list(rule["set"].items()):
            if not isinstance(value, str) or value.startswith("/") or "." not in key:
                continue
            loss_name, _, param = key.partition(".")
            entry = losses.get(loss_name)
            uri = entry.get("uri") if isinstance(entry, dict) else entry
            if not isinstance(uri, str) or not uri.startswith("/"):
                continue
            try:
                refs = registry.facts(uri).refs
            except Exception:
                continue
            if (refs or {}).get(param) in REFERENCE_TYPES:
                resolved = resolve_alias(value, aliases)
                if resolved is not None:
                    rule["set"][key] = resolved


class _Resolver:
    def __init__(self, aliases, globals_, provenance, problems):
        self.aliases = aliases
        self.globals = globals_
        self.provenance = provenance
        self.problems = problems

    def walk(self, value, path, block_vars=frozenset()):
        if isinstance(value, dict):
            if len(path) == 3 and path[:2] == ("model", "templates"):
                block_vars = self.template_variables(value, path)
            out = {}
            for key, item in value.items():
                child = path + (key,)
                if key == "uri" and isinstance(item, str):
                    out[key] = self.uri(item, child, block_vars)
                elif isinstance(item, str) and self.short_call(child):
                    out[key] = self.uri(item, child, block_vars)
                else:
                    out[key] = self.walk(item, child, block_vars)
            return self.reference_params(out)
        if isinstance(value, list):
            return [self.walk(item, path + (index,), block_vars) for index, item in enumerate(value)]
        if isinstance(value, str):
            return self.substitute(value, path, block_vars)
        return value

    def reference_params(self, out):
        """A lego call's params that name another lego by alias (refs of a lego type), written as URIs so that
        resolved.yaml stands alone; names that resolve to nothing stay for check to report."""
        uri = out.get("uri")
        params = out.get("params")
        if not isinstance(uri, str) or not uri.startswith("/") or not isinstance(params, dict):
            return out
        try:
            refs = registry.facts(uri).refs
        except Exception:
            return out
        for param, ref_type in (refs or {}).items():
            value = params.get(param)
            if ref_type in REFERENCE_TYPES and isinstance(value, str) and not value.startswith("/"):
                resolved = resolve_alias(value, self.aliases)
                if resolved is not None:
                    params[param] = resolved
        return out

    def template_variables(self, template, path):
        """The variable names of a model template; inside the template body they stay literal for cirak and win
        over a param of the same name, outside it the param applies."""
        declared = template.get("variables") if isinstance(template.get("variables"), dict) else {}
        return frozenset(str(name) for name in declared)

    def short_call(self, path):
        if path in SHORT_CALL_PATHS:
            return True
        return len(path) >= 2 and path[-1] in INIT_ROLES and "init" in path[:-1]

    def uri(self, text, path, block_vars=frozenset()):
        text = self.substitute(text, path, block_vars)
        if not isinstance(text, str):
            return text
        resolved = resolve_alias(text, self.aliases)
        if resolved is None:
            self.problems.append(error("unknown_alias", f"{text!r} is not a known alias and does not start with /",
                                       self.provenance.get(path),
                                       hint="include an alias pack such as /alias/kalfa/tabular or write the full URI"))
            return text
        return resolved

    def substitute(self, text, path, block_vars=frozenset()):
        if "$" not in text:
            return text

        def lookup(name):
            base, _, field_name = name.partition(".")
            if base in BUILTIN_VARIABLES or base in block_vars:
                return True, None
            if base not in self.globals:
                self.problems.append(error("unknown_variable", f"unknown variable ${name}$",
                                           self.provenance.get(path), hint="define it under params"))
                return True, None
            holder = self.globals[base]
            if not field_name:
                return False, holder
            if not isinstance(holder, dict) or field_name not in holder:
                self.problems.append(error("unknown_variable", f"param {base!r} has no field {field_name!r}",
                                           self.provenance.get(path)))
                return True, None
            return False, holder[field_name]

        whole = TOKEN.fullmatch(text)
        if whole and whole.group(1):
            keep, value = lookup(whole.group(1))
            return text if keep else value

        def piece(match):
            if match.group(1) is None:
                return "$"
            keep, value = lookup(match.group(1))
            return match.group(0) if keep else str(value)

        return TOKEN.sub(piece, text)


def written_config(data):
    """The resolved config as resolved.yaml writes it: no alias table, everything else as loaded."""
    return {key: value for key, value in data.items() if key != "alias"}


def problems_text(problems):
    from cirak.errors import render_problems

    return render_problems(list(problems))


__all__ = ["Problem", "Surface", "TOP_KEYS", "load_surface", "parse_param", "parse_set", "parse_sets",
           "resolve_alias", "written_config"]
