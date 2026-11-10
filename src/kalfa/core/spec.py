import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.constructor import DuplicateKeyError
from ruamel.yaml.error import YAMLError

from .errors import ProblemsError, error, warning


LOADER_KEYS = ("include", "params", "plugins")
PARAM_FLAG = "-p"
TOKEN = re.compile(r"\$\$|\$([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)\$")
BUILTINS = ("datetime",)
DATETIME_FORMAT = "%Y%m%d_%H%M%S"
MISSING = object()


@dataclass
class Layer:
    label: str
    data: dict
    file: str | None = None
    includes: list = field(default_factory=list)


@dataclass
class Config:
    data: dict
    params: dict
    plugins: list
    layers: list
    directories: list
    overrides: dict
    problems: list = field(default_factory=list)

    @property
    def errors(self):
        return [problem for problem in self.problems if problem.severity == "error"]

    @property
    def ok(self):
        return not self.errors


def is_call(value):
    return isinstance(value, dict) and "use" in value


def join(path, key):
    if isinstance(key, int) and not isinstance(key, bool):
        return f"{path}[{key}]"
    return f"{path}.{key}" if path else str(key)


def kind(value):
    return type(value).__name__


def scalar(value):
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):
        return str(value)
    if isinstance(value, date):
        return value.isoformat()
    return value


def plain(value):
    if isinstance(value, dict):
        return {scalar(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return scalar(value)


def clone(value):
    if isinstance(value, dict):
        return {key: clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone(item) for item in value]
    return value


def listed(value):
    if value is None:
        return []
    if isinstance(value, (str, Path, dict)):
        return [value]
    return list(value)


def describe(exc):
    parts = [str(part).strip() for part in (getattr(exc, "context", None), getattr(exc, "problem", None)) if part]
    mark = getattr(exc, "problem_mark", None) or getattr(exc, "context_mark", None)
    line = mark.line + 1 if mark is not None else None
    return ": ".join(parts) or str(exc).strip(), line


def located(path, exc):
    message, line = describe(exc)
    where = f"{path}:{line}" if line else str(path)
    return f"{where}: {message}"


def parse_value(text):
    if isinstance(text, str) and text.lstrip().startswith("@"):
        return text.strip()
    return plain(YAML(typ="safe").load(text))


def assignment(text, code, flag, what):
    expected = f"{flag} expects {what}=VALUE, got {text!r}"
    if not isinstance(text, str):
        raise ProblemsError([error(code, expected)])
    path, separator, raw = text.partition("=")
    path = path.strip()
    if not separator or not path:
        raise ProblemsError([error(code, expected)])
    try:
        value = parse_value(raw.strip())
    except YAMLError as exc:
        message, _ = describe(exc)
        raise ProblemsError([error(code, f"{flag} {path}: cannot parse value {raw!r}: {message}", path)]) from None
    return path, value


def parse_set(text):
    return assignment(text, "bad_set", "--set", "PATH")


def parse_param(text):
    return assignment(text, "bad_param", "-p", "NAME")


def read_yaml(path):
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ProblemsError([error("file_not_found", f"cannot read {path}: {exc.strerror or exc}")]) from None
    except UnicodeDecodeError as exc:
        raise ProblemsError([error("yaml_error", f"{path}: not utf-8 text ({exc.reason})")]) from None
    try:
        loaded = YAML(typ="safe").load(text)
    except DuplicateKeyError as exc:
        raise ProblemsError([error("duplicate_key", located(path, exc))]) from None
    except YAMLError as exc:
        raise ProblemsError([error("yaml_error", located(path, exc))]) from None
    return plain(loaded)


def merge(base, incoming, path, problems, overrides, label):
    for key, value in incoming.items():
        child = join(path, key)
        if key not in base:
            base[key] = clone(value)
            continue
        existing = base[key]
        if isinstance(existing, dict) and isinstance(value, dict) and not is_call(value):
            merge(existing, value, child, problems, overrides, label)
            continue
        if existing != value or type(existing) is not type(value):
            overrides[child] = label
        base[key] = clone(value)
    return base


def nested(pairs):
    data = {}
    for path, value in pairs:
        parts = path.split(".")
        target = data
        for part in parts[:-1]:
            if not isinstance(target.get(part), dict):
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value
    return data


class Substitution:
    def __init__(self, params, problems):
        self.raw = params if isinstance(params, dict) else {}
        self.problems = problems
        self.builtins = {"datetime": datetime.now().strftime(DATETIME_FORMAT)}
        self.done = {}
        self.stack = []
        self.used = set()

    def resolved(self):
        return {name: self.param(name, join("params", name)) for name in self.raw}

    def param(self, name, path):
        if name in self.done:
            return self.done[name]
        if name in self.stack:
            chain = " -> ".join([*self.stack[self.stack.index(name):], name])
            self.problems.append(error("param_cycle", f"param {name!r} refers to itself: {chain}", path))
            return MISSING
        self.stack.append(name)
        value = self.walk(self.raw[name], join("params", name))
        self.stack.pop()
        self.done[name] = value
        return value

    def lookup(self, token, path):
        base, _, member = token.partition(".")
        if base in self.raw:
            self.used.add(base)
            holder = self.param(base, path)
            if holder is MISSING:
                return MISSING
        elif base in self.builtins:
            holder = self.builtins[base]
        else:
            self.problems.append(error("unknown_param", f"unknown param ${token}$", path, "define it under params"))
            return MISSING
        if not member:
            return holder
        if isinstance(holder, dict) and member in holder:
            return holder[member]
        self.problems.append(error("unknown_param", f"param {base!r} has no field {member!r} for ${token}$", path))
        return MISSING

    def text(self, value, path):
        if "$" not in value:
            return value
        whole = TOKEN.fullmatch(value)
        if whole and whole.group(1):
            found = self.lookup(whole.group(1), path)
            return value if found is MISSING else clone(found)

        def piece(match):
            if match.group(1) is None:
                return "$"
            found = self.lookup(match.group(1), path)
            return match.group(0) if found is MISSING else str(found)

        return TOKEN.sub(piece, value)

    def walk(self, value, path):
        if isinstance(value, dict):
            return {key: self.walk(item, join(path, key)) for key, item in value.items()}
        if isinstance(value, list):
            return [self.walk(item, join(path, index)) for index, item in enumerate(value)]
        if isinstance(value, str):
            return self.text(value, path)
        return value


def substitute(data, params, problems):
    substitution = Substitution(params, problems)
    substitution.resolved()
    return substitution.walk(data, "")


class Loader:
    def __init__(self):
        self.problems = []
        self.layers = []
        self.directories = []
        self.data = {}
        self.params = {}
        self.plugins = []
        self.overrides = {}
        self.done = set()
        self.param_layers = {}

    def file(self, raw, label, stack, includer=None):
        path = Path(raw).expanduser()
        resolved = str(path.resolve())
        if resolved in stack:
            chain = " -> ".join(Path(entry).name for entry in [*stack[stack.index(resolved):], resolved])
            hint = f"included by {includer}" if includer else None
            self.problems.append(error("include_cycle", f"include cycle: {chain}", hint=hint))
            return
        if resolved in self.done:
            return
        try:
            loaded = read_yaml(path)
        except ProblemsError as exc:
            hint = f"included by {includer}" if includer else None
            self.problems.extend(error(problem.code, problem.message, problem.path, hint) for problem in exc.problems)
            return
        self.done.add(resolved)
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            message = f"{path}: the top level must be a mapping, not {kind(loaded)}"
            self.problems.append(error("not_a_mapping", message))
            return
        directory = str(Path(resolved).parent)
        if directory not in self.directories:
            self.directories.append(directory)
        layer = Layer(label, loaded, resolved)
        self.includes(loaded.get("include"), Path(resolved).parent, [*stack, resolved], layer, label)
        self.layers.append(layer)
        self.apply(layer)

    def includes(self, entries, directory, stack, layer, includer):
        if entries is None:
            return
        if isinstance(entries, str):
            entries = [entries]
        if not isinstance(entries, list):
            self.problems.append(error("bad_include", f"{includer}: include must be a list of file paths", "include"))
            return
        for index, entry in enumerate(entries):
            if not isinstance(entry, str):
                message = f"{includer}: include entries must be file paths, got {entry!r}"
                self.problems.append(error("bad_include", message, join("include", index)))
                continue
            target = Path(entry).expanduser()
            if not target.is_absolute():
                target = directory / target
            layer.includes.append(str(target.resolve()))
            self.file(target, entry, stack, includer)

    def mapping(self, mapping, number):
        if not isinstance(mapping, dict):
            message = f"mapping {number} must be a mapping, not {kind(mapping)}"
            self.problems.append(error("not_a_mapping", message))
            return
        layer = Layer("mapping", plain(mapping))
        self.includes(layer.data.get("include"), Path.cwd(), [], layer, "mapping")
        self.layers.append(layer)
        self.apply(layer)

    def assignments(self, items, parser, code, label, section=None):
        pairs = []
        for item in items:
            if isinstance(item, str):
                try:
                    pairs.append(parser(item))
                except ProblemsError as exc:
                    self.problems.extend(exc.problems)
            elif isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[0], str):
                pairs.append((item[0], item[1]))
            else:
                message = f"{label} expects NAME=VALUE text or a (name, value) pair, got {item!r}"
                self.problems.append(error(code, message))
        if not pairs:
            return
        data = nested(pairs)
        layer = Layer(label, {section: data} if section else data)
        self.layers.append(layer)
        self.apply(layer)

    def apply(self, layer):
        label = layer.label
        params = layer.data.get("params")
        if params is not None:
            if isinstance(params, dict):
                merge(self.params, params, "params", self.problems, self.overrides, label)
                self.param_layers.update(dict.fromkeys(params, label))
            else:
                self.problems.append(error("bad_params", f"{label}: params must be a mapping", "params"))
        plugins = layer.data.get("plugins")
        if isinstance(plugins, str):
            plugins = [plugins]
        if plugins is not None:
            if not isinstance(plugins, list):
                self.problems.append(error("bad_plugins", f"{label}: plugins must be a list of modules", "plugins"))
            else:
                self.pool(plugins, label)
        own = {key: value for key, value in layer.data.items() if key not in LOADER_KEYS}
        merge(self.data, own, "", self.problems, self.overrides, label)

    def pool(self, plugins, label):
        for index, entry in enumerate(plugins):
            if not isinstance(entry, str):
                message = f"{label}: plugin entries must be module names, got {entry!r}"
                self.problems.append(error("bad_plugins", message, join("plugins", index)))
            elif entry not in self.plugins:
                self.plugins.append(entry)

    def finish(self):
        substitution = Substitution(self.params, self.problems)
        params = substitution.resolved()
        data = substitution.walk(self.data, "")
        self.unused(substitution.used)
        return Config(data, params, list(self.plugins), self.layers, self.directories, self.overrides, self.problems)

    def unused(self, used):
        for name in self.params:
            if name in used:
                continue
            path = join("params", name)
            if self.param_layers.get(name) == PARAM_FLAG:
                message = f"{PARAM_FLAG} {name}: nothing in the config reads ${name}$"
                hint = "a param is read as $name$; a leaf is overridden with --set path=value"
                self.problems.append(error("unused_param", message, path, hint))
            else:
                self.problems.append(warning("unused_param", f"param {name!r} is not read by any $...$", path))


def load(paths=(), sets=(), params=(), mappings=()):
    loader = Loader()
    for raw in listed(paths):
        loader.file(raw, str(raw), [])
    for number, mapping in enumerate(listed(mappings), 1):
        loader.mapping(mapping, number)
    loader.assignments(listed(sets), parse_set, "bad_set", "--set")
    loader.assignments(listed(params), parse_param, "bad_param", PARAM_FLAG, "params")
    return loader.finish()
