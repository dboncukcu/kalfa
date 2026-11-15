import difflib
import importlib
import importlib.util
import inspect
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .errors import BuildError, error


LOGGER = logging.getLogger("kalfa.aliases")
HINT = "kalfa legos lists the names"


@dataclass
class Entry:
    name: str
    target: str
    slot: str | None = None
    doc: str = ""
    params: dict = field(default_factory=dict)
    examples: list = field(default_factory=list)
    plugin: bool = False


ALIASES = {}
SEARCH_DIRS = []


def target_path(target):
    if isinstance(target, str):
        return target
    if inspect.ismodule(target):
        return target.__name__
    module = getattr(target, "__module__", None)
    qualname = getattr(target, "__qualname__", None) or getattr(target, "__name__", None)
    if module and qualname:
        return f"{module}.{qualname}"
    return repr(target)


def imports_as(path, target):
    try:
        return import_target(path) is target
    except BuildError:
        return False


def same_target(first, second):
    if first is second:
        return True
    if isinstance(first, str) != isinstance(second, str):
        path, target = (first, second) if isinstance(first, str) else (second, first)
        if imports_as(path, target):
            return True
    if not isinstance(first, str) and not isinstance(second, str):
        if "<" in target_path(first) or "<" in target_path(second):
            return False
    return target_path(first) == target_path(second)


def register(name, target, *, slot=None, doc="", params=None, examples=None, plugin=False):
    if not isinstance(name, str) or not name:
        raise BuildError(f"an alias name must be a non empty string, got {name!r}")
    if target is None:
        raise BuildError(f"{name!r}: the target must be an import path or an object")
    entry = Entry(name, target, slot, doc or "", dict(params or {}), list(examples or []), bool(plugin))
    existing = ALIASES.get(name)
    if existing is not None:
        if same_target(existing.target, target):
            if doc and not existing.doc:
                existing.doc = doc
            if slot and not existing.slot:
                existing.slot = slot
            return existing
        if existing.plugin and plugin:
            LOGGER.warning("alias %r moves from %s to %s", name, target_path(existing.target), target_path(target))
        else:
            raise BuildError(f"{name!r} is already registered as {target_path(existing.target)}; "
                             f"{target_path(target)} cannot take the name ({HINT})")
    ALIASES[name] = entry
    return entry


def annotate(name, params=None, examples=None):
    entry = ALIASES.get(name) if isinstance(name, str) else None
    if entry is None:
        raise BuildError(f"{name!r} is not a registered name, so it takes no notes ({HINT})")
    entry.params.update(params or {})
    for example in examples or ():
        if example not in entry.examples:
            entry.examples.append(example)
    return entry


def lego(name, *, slot=None, doc="", params=None, examples=None):
    def decorate(target):
        text = doc or (getattr(target, "__doc__", None) or "").strip()
        register(name, target, slot=slot, doc=text, params=params, examples=examples, plugin=True)
        return target
    return decorate


def is_import_path(text):
    return isinstance(text, str) and "." in text and not text.startswith(".")


def resolve(name):
    entry = ALIASES.get(name) if isinstance(name, str) else None
    if entry is not None:
        return entry.target
    if is_import_path(name):
        return name
    raise BuildError(f"{name!r} is not a registered name and not an import path such as pkg.module.Class ({HINT})")


def suggestion(name):
    close = difflib.get_close_matches(str(name), list(ALIASES), n=3, cutoff=0.6)
    return "did you mean " + ", ".join(close) if close else None


def add_search_dir(directory):
    if directory and str(directory) not in SEARCH_DIRS:
        SEARCH_DIRS.append(str(directory))
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))


def missing_prefix(exc, module_name):
    missing = getattr(exc, "name", None)
    return bool(missing) and (missing == module_name or module_name.startswith(missing + "."))


def import_target(path):
    if not isinstance(path, str):
        return path
    if not path or path.startswith(".") or path.endswith("."):
        raise BuildError(f"{path!r} is not an import path such as pkg.module.Class")
    parts = path.split(".")
    if parts[0] not in sys.modules:
        importlib.invalidate_caches()
    failure = None
    for cut in range(len(parts), 0, -1):
        module_name = ".".join(parts[:cut])
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if not missing_prefix(exc, module_name):
                raise BuildError(f"{path}: importing {module_name} failed: {exc}") from exc
            failure = exc
            continue
        except Exception as exc:
            raise BuildError(f"{path}: importing {module_name} failed: {type(exc).__name__}: {exc}") from exc
        target = module
        for attribute in parts[cut:]:
            try:
                target = getattr(target, attribute)
            except AttributeError as exc:
                raise BuildError(f"{path}: {target_path(target)} has no attribute {attribute!r}") from exc
        return target
    raise BuildError(f"{path}: {failure}")


def plugin_file(name):
    candidate = Path(name)
    if candidate.suffix != ".py":
        return None
    roots = [Path.cwd(), *[Path(directory) for directory in SEARCH_DIRS]]
    if candidate.is_absolute():
        roots = [None]
    for root in roots:
        path = candidate if root is None else root / candidate
        if path.is_file():
            return path.resolve()
    return None


def import_file(path):
    name = path.stem
    found = sys.modules.get(name)
    if found is not None and getattr(found, "__file__", None) and Path(found.__file__).resolve() == path:
        return found
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def import_plugins(names, problems):
    loaded = []
    importlib.invalidate_caches()
    for name in names or ():
        try:
            path = plugin_file(str(name))
            if path is not None:
                loaded.append(import_file(path))
            elif isinstance(name, str) and name.endswith(".py"):
                raise ImportError(f"no file {name} under the working directory or the config directories")
            else:
                loaded.append(importlib.import_module(str(name)))
        except Exception as exc:
            problems.append(error("plugin_import", f"{name}: {type(exc).__name__}: {exc}", "plugins",
                                  "a plugin is a module on sys.path or a .py file next to the config"))
    return loaded


def catalog(slot=None, word=None):
    needle = word.lower() if isinstance(word, str) and word else None
    found = []
    for name in sorted(ALIASES):
        entry = ALIASES[name]
        if slot is not None and entry.slot != slot:
            continue
        if needle is not None and needle not in name.lower() and needle not in entry.doc.lower():
            continue
        found.append(entry)
    return found


def entry_of(name):
    return ALIASES.get(name)

