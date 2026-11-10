import inspect
import re
from pathlib import Path

from kalfa.commands.guide.words import and_list, unique, value_text
from kalfa.core import aliases, spec
from kalfa.core.build import signature_of
from kalfa.core.refs import SINGULAR


TABLES = ("models", "losses", "metrics", "optimizers", "plots", "rules")
LAYER_LABELS = ("--set", "-p", "mapping")
INDEX = re.compile(r"(\[\d+\])+$")
PARTS = re.compile(r"[.\[]")
DRAWN = {"log": "log uniform", "float": "uniform", "int": "integers"}
STEPPED = {"log": "values in geometric steps", "float": "values in even steps", "int": "integers in even steps"}


def raw_of(config):
    merged = {}
    for layer in getattr(config, "layers", None) or ():
        if isinstance(layer.data, dict):
            spec.merge(merged, layer.data, "", [], {}, layer.label)
    return merged


def walk(value, path, found, pick):
    if isinstance(value, dict):
        for key, item in value.items():
            walk(item, spec.join(path, key), found, pick)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            walk(item, spec.join(path, index), found, pick)
    elif isinstance(value, str):
        found.extend((path, item) for item in pick(value))
    return found


def tokens(text):
    return [match.group(1) for match in spec.TOKEN.finditer(text) if match.group(1)]


def tokens_in(value):
    return unique(name for _, name in walk(value, "", [], tokens))


def uses_of(raw):
    uses = {}
    for path, name in walk(raw, "", [], tokens):
        uses.setdefault(name.split(".")[0], []).append(INDEX.sub("", path))
    return {name: unique(paths) for name, paths in uses.items()}


def refs_of(data):
    return walk(data, "", [], lambda text: [text] if text.startswith("@") else [])


def ref_name(value, section):
    lead = f"@{section}."
    return value[len(lead):].split(".")[0] if isinstance(value, str) and value.startswith(lead) else None


def ref_meaning(text):
    parts = str(text)[1:].split(".")
    root, rest = parts[0], parts[1:]
    if root in SINGULAR and rest:
        kind, name, inner = SINGULAR[root], rest[0], rest[1:]
        if not inner:
            return f"the {kind} {name}"
        if root == "models" and inner[0] == "ema":
            tail = f" (its {'.'.join(inner[1:])})" if inner[1:] else ""
            return f"the EMA copy of the model {name}{tail}"
        if root == "models":
            return f"the submodule {'.'.join(inner)} of the model {name}"
        return f"the attribute {'.'.join(inner)} of the {kind} {name}"
    if root == "data":
        return f"{'.'.join(rest)}, which the data class exposes once it is fitted" if rest else "the data object"
    if parts == ["generate"]:
        return "the sampler of the generate section"
    if parts == ["train", "step"]:
        return "the step object"
    return f"the node {'.'.join(parts)}"


def holder_of(where):
    parts = [part for part in PARTS.split(where) if part]
    if parts and parts[0] in TABLES + ("train",) and len(parts) > 1:
        return ".".join(parts[:2])
    return parts[0] if parts else where


def quiet(where):
    if where.startswith("optimizers.") and (".over" in where or where.endswith(".minimizes")):
        return True
    return where == "train.predicts" or (where.startswith("rules.") and ".set" in where)


def grid_of(raw):
    from kalfa.commands.sweep.strategies import Grid, strategy_call, strategy_target

    try:
        use, options = strategy_call(raw)
        target = strategy_target(use)
    except Exception:
        return None
    if not (inspect.isclass(target) and issubclass(target, Grid)):
        return None
    try:
        return target(**options)
    except Exception:
        return Grid()


def default_of(target, name):
    if target is None:
        return None
    parameter = signature_of(target).params.get(name)
    if parameter is None or parameter.default is inspect.Parameter.empty:
        return None
    return parameter.default


class Seen:
    def __init__(self, checked, paths=()):
        self.config = checked.config
        self.tree = checked.tree
        self.layers = [layer for layer in getattr(self.config, "layers", None) or () if isinstance(layer.data, dict)]
        self.main = str(Path(paths[0]).expanduser().resolve()) if paths else None
        self.raw = raw_of(self.config)
        data = getattr(self.config, "data", None)
        self.data = data if isinstance(data, dict) else {}
        params = getattr(self.config, "params", None)
        self.params = params if isinstance(params, dict) else {}
        self.uses = uses_of(self.raw)
        self.refs = refs_of(self.data)
        self.grid = grid_of(self.mapping("sweep").get("strategy")) if self.space() else None

    def value(self, *keys):
        current, rest = (self.params, keys[1:]) if keys and keys[0] == "params" else (self.data, keys)
        for key in rest:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        return current

    def mapping(self, *keys):
        found = self.value(*keys)
        return found if isinstance(found, dict) else {}

    def node(self, path):
        return self.tree.nodes.get(path) if self.tree is not None else None

    def target(self, path):
        node = self.node(path)
        return node.target if node is not None else None

    def tabled(self):
        return getattr(self.target("data"), "predictions", True) is not False

    def found(self, use):
        if self.tree is None or not aliases.is_import_path(use):
            return None
        return next((node.target for node in self.tree.nodes.values() if node.use == use and node.target is not None),
                    None)

    def predicts(self):
        models = self.mapping("models")
        asked = self.mapping("train").get("predicts")
        if asked is False:
            return None
        if asked is None:
            return next(iter(models)) if len(models) == 1 else None
        name = ref_name(asked, "models")
        return name if name in models else None

    def first_output(self, model):
        outputs = self.mapping("models", model).get("outputs")
        if isinstance(outputs, str):
            return outputs
        return str(outputs[0]) if isinstance(outputs, list) and outputs else model

    def targets(self):
        data = self.mapping("data")
        targets = data.get("targets")
        if isinstance(targets, dict):
            return [str(name) for name in targets]
        found = aliases.target_path(self.target("data")) if self.target("data") is not None else ""
        if found.endswith(".ImageFolder"):
            return ["label"] if data.get("labels", True) is not False else []
        return ["targets"] if found.endswith(".TextLines") else None

    def minimizers(self, loss):
        return [name for name, item in self.mapping("optimizers").items()
                if isinstance(item, dict) and ref_name(item.get("minimizes"), "losses") == loss]

    def patches(self):
        for name, rule in self.mapping("rules").items():
            if isinstance(rule, dict) and isinstance(rule.get("set"), dict):
                for path, value in rule["set"].items():
                    yield str(name), str(path), value

    def readers(self, path):
        holders = [holder_of(where) for where, text in self.refs
                   if (text[1:] == path or text[1:].startswith(path + ".")) and not quiet(where)]
        return [holder for holder in unique(holders) if holder != path]

    def space(self):
        space = self.mapping("sweep").get("space")
        return space if isinstance(space, dict) else {}


def grid_count(grid, entry):
    try:
        return len(grid.axes({"range": entry})["range"])
    except Exception:
        return None


def range_text(entry, grid=None):
    if isinstance(entry, list):
        return and_list(value_text(item) for item in entry)
    if isinstance(entry, dict) and len(entry) == 1:
        kind, bounds = next(iter(entry.items()))
        if isinstance(bounds, list) and len(bounds) == 2:
            ends = f"from {value_text(bounds[0])} to {value_text(bounds[1])}"
            if grid is None or kind not in STEPPED:
                return f"{DRAWN.get(kind, kind)} {ends}"
            count = grid_count(grid, entry)
            if count == 1:
                return f"only {value_text(bounds[0])}"
            return f"{count} {STEPPED[kind]} {ends}" if count else f"{STEPPED[kind]} {ends}"
    return value_text(entry)


def swept(seen, name):
    space = seen.space()
    if name not in space:
        return None
    entry = space[name]
    verb = "tries" if isinstance(entry, list) or seen.grid is not None else "draws"
    return f"the sweep {verb} {range_text(entry, seen.grid)}"


def param_value_notes(seen, value):
    notes = []
    for name in tokens_in(value):
        base, _, field = name.partition(".")
        if base == "datetime" and base not in seen.params:
            notes.append("$datetime$ is the start time of the run, such as 20261109_143000")
            continue
        if base not in seen.params:
            continue
        found = seen.params[base]
        if field:
            found = found.get(field) if isinstance(found, dict) else None
        sweep = swept(seen, base) if not field else None
        notes.append(f"${name}$ is {value_text(found)}" + (f" here; {sweep}" if sweep else ""))
    return notes


def layer_name(label):
    return label if label in LAYER_LABELS else Path(str(label)).name


def dig(data, keys):
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return False, None
        current = current[key]
    return True, current


def layer_notes(seen, keys, value):
    found = [layer for layer in seen.layers if dig(layer.data, keys)[0]]
    if not found or [layer.file for layer in found] == [seen.main]:
        return []
    names = unique(layer_name(layer.label) for layer in found)
    if dig(found[-1].data, keys)[1] != value:
        return [f"merged from {and_list(names)}"]
    if len(found) > 1:
        return [f"{names[-1]} writes it over {and_list(names[:-1])}"]
    return [f"written by {names[-1]}"]


def override_notes(seen, path):
    notes = []
    for written, label in (getattr(seen.config, "overrides", None) or {}).items():
        if written.startswith((path + ".", path + "[")):
            notes.append(f"{layer_name(label)} writes {written[len(path) + 1:]} over an earlier layer")
    return notes


def patch_text(path, value):
    if isinstance(value, dict) and set(value) == {"times"}:
        return f"multiplies {path} by {value_text(value['times'])}"
    if isinstance(value, dict) and set(value) == {"plus"}:
        return f"adds {value_text(value['plus'])} to {path}"
    return f"sets {path} to {value_text(value)}"


def patch_notes(seen, path):
    notes = []
    for rule, written, value in seen.patches():
        if written == path or written.startswith((path + ".", path + "[")):
            notes.append(f"rule {rule} {patch_text('it' if written == path else written[len(path) + 1:], value)}")
    return notes
