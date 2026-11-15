import inspect

from kalfa.commands.guide.words import flat, lower_first, short, unique
from kalfa.core import aliases


LEGO_KEYS = ("activation", "init", "schedule", "step", "strategy", "tokenizer", "when", "feed", "run", "log", "layers",
             "each", "split", "transform", "train_transform", "sampler")
COLUMNS = ("features", "targets")


def use_of(value):
    if isinstance(value, str) and value and not value.startswith(("@", "$")):
        return value
    if isinstance(value, dict) and isinstance(value.get("use"), str):
        return value["use"]
    return None


def lego_name(text):
    return aliases.entry_of(text) is not None or aliases.is_import_path(text)


def legos_in(value, lego=False):
    found = []
    if isinstance(value, dict):
        use = value.get("use")
        if isinstance(use, str):
            found.append(use)
        for key, item in value.items():
            if key != "use":
                found += legos_in(item, key in LEGO_KEYS or (lego and not isinstance(use, str)))
    elif isinstance(value, list):
        for item in value:
            found += legos_in(item, lego)
    elif lego and isinstance(value, str) and lego_name(value):
        found.append(value)
    return found


def first_line(target):
    own = target.__dict__.get("__doc__") if inspect.isclass(target) else getattr(target, "__doc__", None)
    sentence = flat(own if isinstance(own, str) else "").split(". ")[0].rstrip(".")
    if not sentence or len(sentence) > 140 or "`" in sentence or ":math:" in sentence:
        return None
    return lower_first(sentence)


def import_note(use, target):
    if target is None:
        return f"{use}: an import path"
    kind = "class" if inspect.isclass(target) else "function" if inspect.isroutine(target) else "object"
    name = getattr(target, "__name__", None) or type(target).__name__
    line = first_line(target) if kind != "object" else None
    return f"{use}: an import path to the {kind} {name}" + (f"; {line}" if line else "")


def named_for(use):
    if not aliases.is_import_path(use):
        return None
    found = [entry for entry in aliases.ALIASES.values() if aliases.target_path(entry.target) == use]
    found.sort(key=lambda entry: entry.plugin)
    return found[0] if found else None


def known(use):
    return aliases.entry_of(use) or named_for(use)


def lego_note(use, target=None):
    entry = aliases.entry_of(use)
    if entry is not None:
        doc = lower_first(short(entry.doc))
        if doc and doc.lower() != use.lower():
            return f"{use}: {doc}"
        return f"{use} is {aliases.target_path(entry.target)}"
    twin = named_for(use)
    if twin is not None:
        doc = lower_first(short(twin.doc))
        return f"{use}: the target of the short name {twin.name}" + (f", {doc}" if doc else "")
    if aliases.is_import_path(use):
        return import_note(use, target)
    return f"{use} is not a short name kalfa knows"


def param_lines(use, value, skip=(), lead=""):
    entry = known(use) if use else None
    if entry is None or not isinstance(value, dict):
        return []
    return [f"{lead}{key}: {short(entry.params[key])}" for key in value
            if key != "use" and key not in skip and key in entry.params]


def call_notes(seen, path, value, skip=()):
    use = use_of(value)
    if use is None:
        return []
    notes = [lego_note(use, seen.target(path))] + param_lines(use, value, skip)
    inner = value.items() if isinstance(value, dict) else ()
    nested = [name for key, item in inner if key != "use" and key not in skip
              for name in legos_in(item, key in LEGO_KEYS)]
    return notes + [lego_note(name, seen.found(name)) for name in unique(nested) if name != use]
