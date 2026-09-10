import difflib
import inspect

from cirak.errors import error, warning

from ..contract import Contract
from ..kinds import kalfa_kind
from .data import DataRules
from .refs import RefRules
from .sections import SectionRules


class Checker(SectionRules, DataRules, RefRules):
    def __init__(self, surface, registry, contract=None):
        self.surface = surface
        self.data = surface.data
        self.raw = surface.raw
        self.registry = registry
        self.contract = contract or Contract.load()
        self.sets = self.contract.sets
        self.history_sets = {prefix: name for name, prefix in self.contract.history_prefix.items()}
        self.problems = []
        self.templates = {}
        self.models = {}
        self.trained = []
        self.composites = []
        self.optimizers = {}
        self.losses = {}
        self.metrics = {}
        self.preprocessors = {}
        self.present = {"train": True, "valid": None, "test": None}
        self.structural = 0
        self.header = None
        self.weight_checks = []
        self.target_checks = []

    def error(self, kind, message, path=(), hint=None):
        self.problems.append(error(kind, message, self.surface.source(path), hint))

    def warning(self, kind, message, path=(), hint=None):
        self.problems.append(warning(kind, message, self.surface.source(path), hint))

    def failed(self):
        return any(problem.severity == "error" for problem in self.problems)

    def run(self):
        if not self.top():
            self.structural = len(self.problems)
            return
        self.data_section()
        self.model_section()
        self.definitions()
        self.training_section()
        self.plots_section()
        self.figures_section()
        self.generate_section()
        self.sweep_section()
        self.structural = len(self.problems)
        self.data_header()
        self.target_fields_of(self.data.get("training") or {})
        for section, name, entry, path in self.target_checks:
            self.definition_target(section, name, entry, path)
        for name, definition, weights, path in self.weight_checks:
            self.weights_of(name, definition, weights, path)

    def header_only_errors(self):
        return not any(problem.severity == "error" for problem in self.problems[:self.structural])

    def keys(self, mapping, allowed, path, required=()):
        if not isinstance(mapping, dict):
            self.error("invalid_section", f"{'.'.join(map(str, path)) or 'config'} must be a mapping", path)
            return False
        for key in mapping:
            if key not in allowed:
                close = difflib.get_close_matches(str(key), [str(name) for name in allowed], n=1)
                self.error("unknown_key", f"unknown key {key!r} under {'.'.join(map(str, path)) or 'the config'}",
                           path + (key,), hint=f"did you mean {close[0]!r}?" if close else None)
        for key in required:
            if key not in mapping:
                self.error("missing_key", f"{'.'.join(map(str, path + (key,)))} is required", path)
        return True

    def call_of(self, value, path, kinds, what):
        if isinstance(value, str):
            uri = value
        elif isinstance(value, dict) and isinstance(value.get("uri"), str):
            uri = value["uri"]
            if "params" in value and not isinstance(value["params"], dict):
                self.error("invalid_params", f"params of {what} must be a mapping", path)
        else:
            self.error("invalid_call", f"{what} must be a lego call: a short name or {{uri, params}}", path)
            return None
        if not uri.startswith("/"):
            return None
        if self.registry.lookup(uri) is None:
            close = difflib.get_close_matches(uri, self.registry.uris(), n=1)
            self.error("unknown_uri", f"{uri} is not registered", path,
                       hint=f"did you mean {close[0]}?" if close else None)
            return None
        kind = kalfa_kind(uri)
        if kinds and kind is not None and kind not in kinds:
            self.error("kind_mismatch", f"{uri} is a {kind} lego, {what} needs {' or '.join(kinds)}", path)
        return uri

    def fact_of(self, call, name):
        uri = call.get("uri") if isinstance(call, dict) else call
        if not isinstance(uri, str) or self.registry.lookup(uri) is None:
            return None
        return self.registry.facts(uri).get(name)

    def parameters(self, target):
        if target is None:
            return None
        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):
            return None
        if any(parameter.kind is parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
            return None
        return set(signature.parameters)


def sets_text(sizes, loaded=False, sets=("train", "valid", "test")):
    label = "sets (loaded)" if loaded else "sets (before filters, from the file header)"
    if sizes is None:
        return f"{label}: unknown (the data header could not be read)"
    parts = []
    for name in sets:
        size = sizes.get(name)
        parts.append(f"{name} {'?' if size is None else size if size else 'none'}")
    return f"{label}: " + ", ".join(parts)
