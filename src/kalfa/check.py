"""kalfa's own check: the config surface against CONFIG.md, reported in cirak's Problem form.

Structural rules (keys, required sections), lego kinds per position, references, rule set targets and values,
monitors, the field globs against the data header, and the set table. Signature checks of the compiled recipe
come from cirak, binding problems from tezgah; both are appended by the api.
"""

import difflib
import inspect
import math
from pathlib import Path

from cirak.errors import error, warning
from cirak.registry import registry as default_registry

from .driver import MODEL_KEYS, is_composite, is_shortcut, models_of
from .kinds import kalfa_kind, RESERVED_BLOCKS, SETS, TRAINING_FIXED
from .std.pre import RESERVED_FEATURES, RESERVED_INPUT, assign_fields, torch_dtype
from .std.runtime import expand_targets
from .std.source import STREAM_SOURCES
from .std.source import header as read_header
from .std.split import sizes as split_sizes

TOP_KEYS = ("plugins", "params", "seed", "device", "data", "model", "metrics", "losses", "optimizers", "training",
            "generate", "plots", "sweep", "record", "alias")
TOP_REQUIRED = ("data", "model", "losses", "training", "record")
DATA_KEYS = ("source", "filter", "split", "batch", "preprocessors", "drop", "fields", "feed")
DATA_REQUIRED = ("source", "split", "batch", "fields", "feed")
RATIO_SPLITS = ("/split/kalfa/random", "/split/kalfa/sequential")
BATCH_KEYS = ("size", "eval_size", "shuffle", "drop_last", "workers", "collate", "balanced", "buffer")
MODEL_SECTION_KEYS = ("templates", "models")
TEMPLATE_KEYS = ("variables", "inputs", "outputs", "nodes")
NODE_KEYS = ("uri", "template", "model", "params", "inputs", "outputs", "init", "repeat")
COMPOSITE_FORBIDDEN = ("optimizer", "init", "ema", "trainable", "weights")
ENTRY_KEYS = ("uri", "params", "every", "sets", "output", "target")
OPTIMIZER_KEYS = ("uri", "params", "loss", "schedule")
RULE_KEYS = ("name", "when", "set", "after")
PLOT_KEYS = ("uri", "params", "inputs")
PRE_KEYS = ("uri", "params", "sets")
FIELD_KEYS = ("preprocessors", "target")
INIT_KEYS = ("weights", "bias", "scale", "patterns")
REPORTS = ("best", "last")
BEST_URI = "/checkpoint/kalfa/best"
HISTORY_SETS = {"train": "train", "val": "valid", "test": "test"}


def check_surface(surface, registry=None):
    """Every problem kalfa finds on the config surface."""
    checker = Checker(surface, registry if registry is not None else default_registry)
    checker.run()
    return checker.problems


def set_table(surface, registry=None):
    """The set sizes the split produces, from the data header; None when the header cannot be read."""
    checker = Checker(surface, registry if registry is not None else default_registry)
    return checker.sizes()


class Checker:
    def __init__(self, surface, registry):
        self.surface = surface
        self.data = surface.data
        self.raw = surface.raw
        self.registry = registry
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
        self.header = None
        self.weight_checks = []
        self.target_checks = []

    def error(self, kind, message, path=(), hint=None):
        self.problems.append(error(kind, message, self.surface.source(path), hint))

    def warning(self, kind, message, path=(), hint=None):
        self.problems.append(warning(kind, message, self.surface.source(path), hint))

    @property
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
        """True when every error so far came from the file stage (data header, weights runs), so the recipe can
        still be shaped."""
        return not any(problem.severity == "error" for problem in self.problems[:getattr(self, "structural", 0)])

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
        """A lego call position: {uri, params} or a string; returns the uri or None after reporting."""
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

    def top(self):
        """The top level keys; False when the sections cannot be walked."""
        if not self.keys(self.data, TOP_KEYS, (), TOP_REQUIRED):
            return False
        if self.data.get("seed") is None:
            self.warning("no_seed", "seed is not written; torch runs unseeded and two runs differ", ("seed",))
        if self.data.get("device") is not None:
            self.call_of(self.data["device"], ("device",), ("device",), "device")
        record = self.data.get("record")
        if "record" in self.data and not isinstance(record, str):
            self.error("invalid_value", "record must be a path", ("record",))
        usable = True
        for key in ("data", "model", "training"):
            if key not in self.data:
                usable = False
            elif not isinstance(self.data[key], dict):
                self.error("invalid_section", f"{key} must be a mapping", (key,))
                usable = False
        return usable

    def data_section(self):
        data = self.data["data"]
        if not self.keys(data, DATA_KEYS, ("data",), DATA_REQUIRED):
            return
        if "source" in data:
            self.call_of(data["source"], ("data", "source"), ("source",), "data.source")
        for position, item in enumerate(data.get("filter") or []):
            path = ("data", "filter", position)
            if isinstance(item, dict):
                self.keys(item, ("query", "sets"), path, ("query",))
                self.sets_of(item.get("sets"), path)
            elif not isinstance(item, str):
                self.error("invalid_value", "a filter is a query string or {query, sets}", path)
        split = data.get("split")
        if isinstance(split, dict) and "uri" in split:
            self.call_of(split, ("data", "split"), ("split",), "data.split")
            self.present = {"train": True, "valid": None, "test": None}
        if isinstance(split, dict) and "uri" in split:
            self.present.update(self.split_presence(split))
        elif isinstance(split, dict):
            self.keys(split, ("ratios", "seed"), ("data", "split"), ("ratios",))
            ratios = split.get("ratios")
            if not (isinstance(ratios, list) and len(ratios) == 3
                    and all(isinstance(part, (int, float)) and not isinstance(part, bool) for part in ratios)):
                self.error("invalid_value", "split.ratios must be three numbers (train, valid, test)",
                           ("data", "split"))
            else:
                self.present = {name: ratio > 0 for name, ratio in zip(SETS, ratios)}
                if not self.present["train"]:
                    self.error("invalid_value", "the train ratio must be positive", ("data", "split"))
        elif "split" in data:
            self.error("invalid_call", "data.split must be {ratios, seed} or {uri, params}", ("data", "split"))
        batch = data.get("batch")
        if isinstance(batch, dict):
            self.keys(batch, BATCH_KEYS, ("data", "batch"), ("size",))
        elif "batch" in data and (not isinstance(batch, int) or isinstance(batch, bool) or batch <= 0):
            self.error("invalid_value", "data.batch must be a positive size or a mapping with size",
                       ("data", "batch"))
        preprocessors = data.get("preprocessors") or {}
        if isinstance(preprocessors, dict):
            for name, entry in preprocessors.items():
                path = ("data", "preprocessors", name)
                if isinstance(entry, dict):
                    self.keys(entry, PRE_KEYS, path, ("uri",))
                    if "sets" in entry:
                        self.sets_of(entry.get("sets"), path)
                self.call_of(entry, path, ("pre",), f"data.preprocessors.{name}")
            self.preprocessors = dict(preprocessors)
        else:
            self.error("invalid_section", "data.preprocessors must be a mapping", ("data", "preprocessors"))
        drop = data.get("drop") or []
        if not isinstance(drop, list):
            self.error("invalid_value", "data.drop must be a list of columns", ("data", "drop"))
        fields = data.get("fields")
        used = set()
        if isinstance(fields, dict):
            for name, spec in fields.items():
                path = ("data", "fields", name)
                if spec is None:
                    spec = {}
                if not self.keys(spec, FIELD_KEYS, path):
                    continue
                chain = spec.get("preprocessors") or []
                if not isinstance(chain, list):
                    self.error("invalid_value", f"fields.{name}.preprocessors must be a list", path)
                    chain = []
                for pre in chain:
                    if pre not in self.preprocessors:
                        self.error("unresolved_ref", f"field {name!r} names preprocessor {pre!r}, "
                                                     f"which data.preprocessors does not define", path)
                    used.add(pre)
                if "target" in spec and not isinstance(spec["target"], bool):
                    self.error("invalid_value", f"fields.{name}.target must be a boolean", path)
                if name == RESERVED_INPUT:
                    self.error("reserved_field", f"{RESERVED_INPUT!r} is a reserved field name", path)
                if name == RESERVED_FEATURES and not spec.get("target"):
                    self.error("reserved_field", f"{RESERVED_FEATURES!r} is reserved for the feature tensor", path)
        elif "fields" in data:
            self.error("invalid_section", "data.fields must be a mapping", ("data", "fields"))
        for name, entry in self.preprocessors.items():
            if isinstance(entry, dict) and isinstance(entry.get("uri"), str):
                self.refs_of(entry["uri"], entry.get("params"), ("data", "preprocessors", name))
                for param, ref_type in self.registry.facts(entry["uri"]).refs.items():
                    if ref_type in ("pre", "preprocessor") and isinstance((entry.get("params") or {}).get(param), str):
                        used.add(entry["params"][param])
        for name in self.preprocessors:
            if name not in used:
                self.warning("unused_preprocessor", f"preprocessor {name!r} is defined but no field uses it",
                             ("data", "preprocessors", name))
        if "feed" in data:
            self.call_of(data["feed"], ("data", "feed"), ("feed",), "data.feed")
        source = data.get("source")
        if isinstance(source, dict) and source.get("uri") in STREAM_SOURCES:
            self.lazy_rules(data)

    def lazy_rules(self, data):
        """What the lazy set refuses (CONFIG.md section 3): a shuffled or folded split, the balanced sampler, the window
        feed, counting components."""
        hint = "the lazy set streams the table: split with sequential or given, feed with table, no balanced " \
               "sampler and no class_weights"
        split = data.get("split")
        if isinstance(split, dict) and ("uri" not in split or split.get("uri") == "/split/kalfa/kfold"):
            self.error("lazy_split", "a stream source cannot be shuffled or folded", ("data", "split"), hint=hint)
        batch = data.get("batch")
        if isinstance(batch, dict) and batch.get("balanced"):
            self.error("lazy_batch", "a stream source cannot be counted for the balanced sampler",
                       ("data", "batch"), hint=hint)
        feed = data.get("feed")
        feed_uri = feed.get("uri") if isinstance(feed, dict) else feed
        if feed_uri == "/feed/kalfa/window":
            self.error("lazy_feed", "the window feed needs the table in memory", ("data", "feed"), hint=hint)
        for name, entry in (self.data.get("losses") or {}).items():
            if _mentions(entry, "/data/kalfa/class_weights"):
                self.error("lazy_data", f"losses.{name} builds class_weights, which counts the train set",
                           ("losses", name), hint=hint)
        for name, entry in (data.get("preprocessors") or {}).items():
            uri = entry.get("uri") if isinstance(entry, dict) else entry
            target = self.registry.resolve_quietly(uri) if isinstance(uri, str) else None
            if target is None:
                continue
            try:
                obj = target(**((entry.get("params") if isinstance(entry, dict) else None) or {}))
            except Exception:
                continue
            if hasattr(obj, "fit") and not hasattr(obj, "partial_fit"):
                self.warning("lazy_fit", f"preprocessor {name!r} ({uri}) has no partial_fit; on a stream its column "
                                         f"is collected in memory to fit", ("data", "preprocessors", name),
                             hint="scalers fit incrementally; one_hot and label_encoder collect the column")

    def split_presence(self, split):
        """Which sets a split lego produces, read from its params where the shape says so."""
        params = split.get("params") or {}
        uri = split.get("uri")
        if uri in RATIO_SPLITS and isinstance(params.get("ratios"), list) and len(params["ratios"]) == 3:
            return {name: ratio > 0 for name, ratio in zip(SETS, params["ratios"])}
        if uri == "/split/kalfa/kfold":
            return {"train": True, "valid": bool(params.get("val")), "test": True}
        if uri == "/split/kalfa/given":
            return {"train": True, "valid": params.get("valid") is not None, "test": params.get("test") is not None}
        return {}

    def column_refs(self):
        """The (column name, path) pairs the data legos reference with a column typed param."""
        data = self.data.get("data") or {}
        found = []
        calls = [("split", data.get("split")), ("feed", data.get("feed"))]
        for name, entry in (data.get("preprocessors") or {}).items():
            calls.append((f"preprocessors.{name}", entry))
        for label, call in calls:
            if not isinstance(call, dict) or not isinstance(call.get("uri"), str):
                continue
            refs = self.registry.facts(call["uri"]).refs
            for param, ref_type in refs.items():
                value = (call.get("params") or {}).get(param)
                if ref_type == "column" and isinstance(value, str):
                    found.append((value, ("data", *label.split("."), "params", param)))
        return found

    def sets_of(self, sets, path):
        if sets is None:
            return
        if not isinstance(sets, list) or any(item not in SETS for item in sets):
            self.error("invalid_value", f"sets must list some of {list(SETS)}", path)

    def model_section(self):
        section = self.data["model"]
        if is_shortcut(section) and any(key in section for key in MODEL_SECTION_KEYS):
            self.error("ambiguous_model", "model mixes the single model shortcut with templates or models",
                       ("model",))
            return
        if is_shortcut(section):
            self.keys(section, MODEL_KEYS, ("model",), ("inputs", "outputs", "nodes"))
            base = ("model",)
        else:
            self.keys(section, MODEL_SECTION_KEYS, ("model",), ("models",))
            base = None
        self.templates, self.models = models_of(section)
        for name, template in self.templates.items():
            path = ("model", "templates", name)
            if name in RESERVED_BLOCKS:
                self.error("model_name_reserved", f"template name {name!r} is reserved for a template block", path)
            if self.keys(template, TEMPLATE_KEYS, path, ("nodes",)):
                variables = template.get("variables")
                if variables is not None and not (isinstance(variables, dict) and all(
                        isinstance(spec, dict) and ("required" in spec or "default" in spec)
                        for spec in variables.values())):
                    self.error("invalid_value", f"variables of template {name!r} map names to {{required: true}} "
                                                f"or {{default: ...}}", path + ("variables",))
                self.nodes_of(template.get("nodes"), path, composite_allowed=False)
        if not self.models:
            self.error("missing_key", "model.models needs at least one model", ("model",))
        for name, definition in self.models.items():
            path = base if base is not None else ("model", "models", name)
            if "." in name:
                self.error("model_name_dot", f"model name {name!r} contains a dot", path)
            if name in RESERVED_BLOCKS:
                self.error("model_name_reserved", f"model name {name!r} is reserved for a template block", path,
                           hint=f"reserved names: {list(RESERVED_BLOCKS)}")
            if name in self.templates:
                self.error("model_name_reserved", f"model {name!r} has the name of a template", path)
            if base is None and not self.keys(definition, MODEL_KEYS, path, ("inputs", "outputs", "nodes")):
                continue
            composite = is_composite(definition)
            if composite:
                self.composites.append(name)
                for key in COMPOSITE_FORBIDDEN:
                    if key in definition:
                        self.error("composite_key", f"composite model {name!r} cannot write {key}", path + (key,))
            else:
                self.trained.append(name)
            self.nodes_of(definition.get("nodes"), path, composite_allowed=True)
            self.init_of(definition.get("init"), path + ("init",))
            if "ema" in definition and not (isinstance(definition["ema"], dict) and "decay" in definition["ema"]):
                self.error("invalid_value", f"ema of model {name!r} must be {{decay: ...}}", path + ("ema",))
            if "trainable" in definition and not isinstance(definition["trainable"], bool):
                self.error("invalid_value", f"trainable of model {name!r} must be a boolean", path + ("trainable",))
            weights = definition.get("weights")
            if weights is not None:
                if not (isinstance(weights, dict) and all(key in weights for key in ("run", "model", "which"))):
                    self.error("invalid_value", f"weights of model {name!r} must be {{run, model, which}}",
                               path + ("weights",))
                else:
                    self.weight_checks.append((name, definition, weights, path + ("weights",)))
                if definition.get("init") is not None:
                    self.error("weights_init", f"model {name!r} writes both weights and init", path)
            optimizer = definition.get("optimizer")
            if isinstance(optimizer, str):
                if optimizer not in (self.data.get("optimizers") or {}):
                    self.error("unresolved_ref", f"model {name!r} uses optimizer {optimizer!r}, "
                                                 f"which optimizers does not define", path + ("optimizer",))
            elif optimizer is not None:
                self.call_of(optimizer, path + ("optimizer",), ("optimizer",), f"optimizer of model {name!r}")
            elif not composite and definition.get("trainable", True):
                self.warning("untrained_model", f"model {name!r} writes no optimizer and is never trained", path)

    def weights_of(self, name, definition, weights, path):
        """The source run of a weights spec: its resolved model block must match this model's structure."""
        from .record import read_resolved
        from .std.builder import weights_path

        run = weights.get("run")
        if weights.get("which") not in ("best", "last", "final"):
            self.error("invalid_value", f"weights.which of model {name!r} must be best, last or final", path)
        resolved = Path(str(run)) / "resolved.yaml"
        if not resolved.exists():
            self.error("weights_run_missing", f"weights of model {name!r}: run {run!r} has no resolved.yaml", path)
            return
        if weights.get("which") in ("best", "last", "final") and not weights_path(weights).exists():
            self.error("weights_missing", f"weights of model {name!r}: {weights_path(weights)} does not exist", path)
        try:
            source = read_resolved(resolved.parent)
        except Exception as exc:
            self.error("weights_run_missing", f"weights of model {name!r}: cannot read {resolved}: {exc}", path)
            return
        _, models = models_of(source.get("model") or {})
        block = models.get(weights.get("model"))
        if block is None:
            self.error("unresolved_ref", f"weights of model {name!r}: run {run!r} has no model "
                                         f"{weights.get('model')!r}; it has {sorted(models)}", path)
            return
        for key in ("inputs", "outputs", "nodes"):
            if block.get(key) != definition.get(key):
                self.error("weights_mismatch", f"weights of model {name!r}: {key} differ from model "
                                               f"{weights.get('model')!r} of run {run!r}", path)

    def nodes_of(self, nodes, path, composite_allowed):
        items = list(enumerate(nodes)) if isinstance(nodes, list) else \
            list(nodes.items()) if isinstance(nodes, dict) else None
        if items is None:
            self.error("invalid_value", "nodes must be a list (a chain) or a mapping (a graph)", path + ("nodes",))
            return
        for key, item in items:
            node_path = path + ("nodes", key)
            if not isinstance(item, dict):
                self.error("invalid_value", "a node is a mapping with uri, template or model", node_path)
                continue
            kinds = [kind for kind in ("uri", "template", "model") if kind in item]
            if len(kinds) != 1:
                self.error("invalid_value", "a node has exactly one of uri, template or model", node_path)
                continue
            self.keys(item, NODE_KEYS, node_path)
            if kinds[0] == "uri":
                self.call_of(item, node_path, ("layer",), "a model node")
                self.init_of(item.get("init"), node_path + ("init",))
            elif kinds[0] == "template":
                if item["template"] not in self.templates:
                    self.error("unresolved_ref", f"template {item['template']!r} is not defined", node_path)
            else:
                if not composite_allowed:
                    self.error("invalid_value", "a template cannot reference a model", node_path)
                elif item["model"] not in self.models or is_composite(self.models[item["model"]]):
                    self.error("unresolved_ref", f"{{model: {item['model']!r}}} names no trained model", node_path)

    def init_of(self, init, path):
        if init is None:
            return
        if isinstance(init, str):
            self.error("invalid_value", "init is a role mapping: {weights, bias, scale, patterns}", path)
            return
        if not self.keys(init, INIT_KEYS, path):
            return
        for role in ("weights", "bias", "scale"):
            if role in init:
                self.call_of(init[role], path + (role,), ("init",), f"init.{role}")
        for position, entry in enumerate(init.get("patterns") or []):
            entry_path = path + ("patterns", position)
            if not (isinstance(entry, dict) and "match" in entry):
                self.error("invalid_value", "an init pattern is {match, weights | bias | scale}", entry_path)
                continue
            for role in ("weights", "bias", "scale"):
                if role in entry:
                    self.call_of(entry[role], entry_path + (role,), ("init",), f"init pattern {role}")

    def definitions(self):
        losses = self.data.get("losses")
        if not isinstance(losses, dict) or not losses:
            self.error("missing_key", "losses needs at least one definition", ("losses",))
            losses = {}
        self.losses = losses
        metrics = self.data.get("metrics") or {}
        if not isinstance(metrics, dict):
            self.error("invalid_section", "metrics must be a mapping", ("metrics",))
            metrics = {}
        self.metrics = metrics
        for name in set(losses) & set(metrics):
            self.error("duplicate_name", f"{name!r} is defined under both losses and metrics", ("metrics", name))
        for section, table, kinds, wrong in (("losses", losses, ("criterion", "objective"), "metric"),
                                             ("metrics", metrics, ("metric", "criterion"), "objective")):
            for name, entry in table.items():
                path = (section, name)
                if not isinstance(entry, dict):
                    entry = {"uri": entry} if isinstance(entry, str) else {}
                if not self.keys(entry, ENTRY_KEYS, path, ("uri",)):
                    continue
                uri = self.call_of(entry, path, kinds, f"{section}.{name}")
                if uri is None:
                    continue
                self.refs_of(uri, entry.get("params"), path)
                facts = self.registry.facts(uri)
                every = entry.get("every", 1)
                if not isinstance(every, int) or isinstance(every, bool) or every < 1:
                    self.error("invalid_value", f"{section}.{name}.every must be a positive integer", path)
                self.sets_of(entry.get("sets"), path)
                if facts.needs_grad and entry.get("sets") != ["train"]:
                    self.error("needs_grad_set", f"{section}.{name} needs gradients; write sets: [train]", path)
                if facts.needs_grad and (self.data.get("training") or {}).get("amp"):
                    names = self.parameters(self.registry.resolve_quietly(uri))
                    if names is not None and "scaler" not in names:
                        self.error("amp_scaler", f"{section}.{name} needs gradients under amp, so its signature "
                                                 f"must take scaler", path)
                if entry.get("output") is not None and not isinstance(entry["output"], str):
                    self.error("invalid_value", f"{section}.{name}.output must be a name", path)
                if entry.get("target") is not None and not is_selector(entry["target"]):
                    self.error("invalid_value", f"{section}.{name}.target must be a field name, a list of names "
                                                f"or a glob", path)
                elif self.compares(uri, facts):
                    self.target_checks.append((section, name, entry, path))
        optimizers = self.data.get("optimizers") or {}
        if not isinstance(optimizers, dict):
            self.error("invalid_section", "optimizers must be a mapping", ("optimizers",))
            optimizers = {}
        users = set()
        for name, definition in self.models.items():
            if isinstance(definition.get("optimizer"), str):
                users.add(definition["optimizer"])
        training = self.data.get("training") or {}
        for name, entry in optimizers.items():
            path = ("optimizers", name)
            if not self.keys(entry, OPTIMIZER_KEYS, path, ("uri",)):
                continue
            self.call_of(entry, path, ("optimizer",), f"optimizers.{name}")
            loss = entry.get("loss", training.get("loss") if len(optimizers) == 1 else None)
            if loss is None:
                self.error("missing_key", f"optimizers.{name}.loss is required", path)
            elif loss not in losses:
                self.error("unresolved_ref", f"optimizers.{name}.loss names {loss!r}, which losses does not define",
                           path)
            if entry.get("schedule") is not None:
                self.call_of(entry["schedule"], path + ("schedule",), ("schedule",), f"optimizers.{name}.schedule")
            if name not in users:
                self.error("unused_optimizer", f"optimizer {name!r} is defined but no model uses it", path)
        self.optimizers = dict(optimizers)
        for name, definition in self.models.items():
            optimizer = definition.get("optimizer")
            if isinstance(optimizer, dict):
                self.optimizers[name] = {"uri": optimizer.get("uri"), "params": optimizer.get("params") or {},
                                         "loss": training.get("loss")}
                if training.get("loss") is None:
                    self.error("missing_key", f"model {name!r} writes an inline optimizer; training.loss names "
                                              f"its loss", ("training",))

    def training_section(self):
        training = self.data["training"]
        if not isinstance(training, dict):
            return
        for key in ("turn", "report"):
            if key not in training:
                self.error("missing_key", f"training.{key} is required", ("training",))
        turn_uri = self.call_of(training["turn"], ("training", "turn"), ("turn",), "training.turn") \
            if "turn" in training else None
        extras = {key: value for key, value in training.items() if key not in TRAINING_FIXED}
        if turn_uri is not None:
            target = self.registry.resolve_quietly(turn_uri)
            names = self.parameters(target)
            accepted = turn_extras(self.registry, turn_uri, target)
            for key in extras:
                if key not in accepted:
                    hint = None if accepted else "the turn declares no extras fact, so it takes no extra keys"
                    self.error("signature_mismatch", f"training.{key} goes to the turn, but {turn_uri} does not "
                                                     f"accept {key!r}", ("training", key), hint=hint)
            if names is not None:
                if "metrics" not in names:
                    self.warning("turn_without_metrics", f"turn {turn_uri} takes no metrics; train/ values are not "
                                                         f"computed", ("training", "turn"))
        has_epochs = training.get("epochs") is not None
        has_steps = training.get("steps") is not None
        if has_epochs == has_steps:
            self.error("missing_key", "training needs exactly one of epochs and steps", ("training",))
        if has_epochs and (not isinstance(training["epochs"], int) or isinstance(training["epochs"], bool)
                           or training["epochs"] < 0):
            self.error("invalid_value", "training.epochs must be a non negative integer", ("training", "epochs"))
        if has_steps and not (isinstance(training["steps"], dict)
                              and all(key in training["steps"] for key in ("total", "turn"))):
            self.error("invalid_value", "training.steps must be {total, turn}", ("training", "steps"))
        if "loss" in training and training["loss"] not in self.losses:
            self.error("unresolved_ref", f"training.loss names {training['loss']!r}, which losses does not define",
                       ("training", "loss"))
        report = training.get("report")
        if "report" in training and report not in REPORTS:
            self.error("invalid_value", "training.report must be best or last", ("training", "report"))
        checkpoint = training.get("checkpoint")
        checkpoint_uri = None
        if checkpoint is not None:
            checkpoint_uri = self.call_of(checkpoint, ("training", "checkpoint"), ("checkpoint",),
                                          "training.checkpoint")
            self.monitors_of(checkpoint, ("training", "checkpoint"), strict=True)
        if report == "best" and checkpoint_uri != BEST_URI:
            self.error("report_mismatch", "report: best needs checkpoint: {uri: best, ...}",
                       ("training", "report"))
        stop = training.get("stop") or []
        if not isinstance(stop, list):
            self.error("invalid_value", "training.stop must be a list of triggers", ("training", "stop"))
            stop = []
        for position, trigger in enumerate(stop):
            path = ("training", "stop", position)
            self.call_of(trigger, path, ("trigger",), f"training.stop[{position}]")
            self.monitors_of(trigger, path, strict=True)
        self.rules_of(training.get("rules") or [])
        self.predicts_of(training)
        self.targets_of(training)
        self.order_of(training, turn_uri)

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

    def monitors_of(self, call, path, strict):
        params = call.get("params") if isinstance(call, dict) else None
        monitor = (params or {}).get("monitor")
        if monitor is None:
            return
        if not isinstance(monitor, str) or "/" not in monitor:
            self.error("invalid_value", f"monitor must be <set>/<name>, got {monitor!r}", path)
            return
        prefix, name = monitor.split("/", 1)
        base = name.split("/", 1)[0]
        if prefix not in HISTORY_SETS:
            self.error("invalid_value", f"monitor {monitor!r} must start with train/, val/ or test/", path)
            return
        if strict and prefix == "test":
            self.error("test_monitor", f"{monitor!r}: stop and checkpoint cannot watch the test set", path)
        if base not in self.losses and base not in self.metrics:
            self.error("unresolved_ref", f"monitor {monitor!r} names {base!r}, which neither losses nor metrics "
                                         f"define", path)
        set_name = HISTORY_SETS[prefix]
        if self.present.get(set_name) is False:
            self.error("set_missing", f"monitor {monitor!r} watches the {set_name} set, which the split does not "
                                      f"produce", path)

    def rules_of(self, rules):
        if not isinstance(rules, list):
            self.error("invalid_value", "training.rules must be a list", ("training", "rules"))
            return
        seen = []
        for position, rule in enumerate(rules):
            path = ("training", "rules", position)
            if not isinstance(rule, dict) or not self.keys(rule, RULE_KEYS, path, ("name", "when", "set")):
                continue
            name = rule.get("name")
            if name in seen:
                self.error("duplicate_name", f"rule {name!r} is defined twice", path)
            if isinstance(name, str) and name.startswith("stop_"):
                self.error("invalid_value", f"rule name {name!r} is reserved for stop triggers", path)
            after = rule.get("after")
            if after is not None and after not in seen:
                self.error("unresolved_ref", f"rule {name!r} waits for {after!r}, which is no earlier rule", path)
            if isinstance(name, str):
                seen.append(name)
            if "when" in rule:
                uri = self.call_of(rule["when"], path + ("when",), ("trigger",), f"rule {name!r} when")
                if isinstance(rule["when"], dict):
                    self.refs_of(uri, rule["when"].get("params"), path + ("when",))
                self.monitors_of(rule["when"], path + ("when",), strict=False)
            targets = rule.get("set")
            if not isinstance(targets, dict) or not targets:
                self.error("invalid_value", f"rule {name!r} needs a set mapping", path)
                continue
            for key, value in targets.items():
                self.set_of(key, value, path + ("set", key))

    def set_of(self, key, value, path):
        owner, _, param = key.partition(".")
        if not param:
            if key != "loss":
                self.error("set_target", f"set target {key!r} is not <opt>.loss, loss, <loss>.<param>, "
                                         f"<opt>.<param> or <model>.trainable", path)
            elif len(self.optimizers) != 1:
                self.error("set_target", "set: {loss: ...} needs exactly one optimizer; write <opt>.loss", path)
            elif value not in self.losses:
                self.error("set_value", f"set loss names {value!r}, which losses does not define", path)
            return
        if param == "loss":
            if owner not in self.optimizers:
                self.error("set_target", f"set target {key!r}: {owner!r} is no optimizer", path)
            elif value not in self.losses:
                self.error("set_value", f"set {key} names {value!r}, which losses does not define", path)
            return
        if param == "trainable" and owner in self.trained:
            if not isinstance(value, bool):
                self.error("set_value", f"set {key} needs a boolean", path)
            return
        if owner in self.optimizers:
            return
        if owner in self.losses:
            entry = self.losses[owner]
            uri = entry.get("uri") if isinstance(entry, dict) else entry
            names = self.parameters(self.registry.resolve_quietly(uri)) if isinstance(uri, str) else None
            if names is not None and param not in names:
                self.error("set_value", f"set {key}: {uri} has no parameter {param!r}", path)
                return
            ref_type = self.registry.facts(uri).refs.get(param) if isinstance(uri, str) else None
            if ref_type is not None and isinstance(value, str):
                from .config import resolve_alias

                if ref_type == "loss" and value not in self.losses:
                    self.error("set_value", f"set {key} names {value!r}, which losses does not define", path)
                elif ref_type != "loss" and resolve_alias(value, self.surface.aliases) is None:
                    self.error("set_value", f"set {key} names {value!r}, which is no known {ref_type}", path)
            return
        self.error("set_target", f"set target {key!r} names neither an optimizer, a loss nor a trained model", path)

    def refs_of(self, uri, params, path):
        """The refs fact of a lego against its written params: lego names must resolve, model and loss names exist."""
        from .config import resolve_alias

        if not isinstance(uri, str) or not isinstance(params, dict):
            return
        for param, ref_type in self.registry.facts(uri).refs.items():
            value = params.get(param)
            if ref_type == "loss" and isinstance(value, dict):
                for name in value:
                    if name not in self.losses:
                        self.error("unresolved_ref", f"{param}: {name!r} is no losses definition",
                                   path + ("params", param))
                continue
            if not isinstance(value, str):
                continue
            if ref_type == "model":
                base = value[:-4] if value.endswith(".ema") else value
                if base not in self.models:
                    self.error("unresolved_ref", f"{param}: {value!r} is no model", path + ("params", param))
            elif ref_type == "loss":
                if value not in self.losses:
                    self.error("unresolved_ref", f"{param}: {value!r} is no losses definition", path + ("params", param))
            elif ref_type in ("pre", "preprocessor"):
                if value not in ((self.data.get("data") or {}).get("preprocessors") or {}):
                    self.error("unresolved_ref", f"{param}: {value!r} is no data.preprocessors definition",
                               path + ("params", param))
            elif ref_type in ("field", "column", "wire", "history", "data"):
                continue
            elif ref_type == "generate" and value == "generate":
                if not isinstance(self.data.get("generate"), (dict, str)):
                    self.error("generate_missing", f"{param}: generate names the generate section, which the config "
                                                   f"does not write", path + ("params", param))
            elif resolve_alias(value, self.surface.aliases) is None:
                self.error("unresolved_ref", f"{param}: {value!r} is no known {ref_type} lego",
                           path + ("params", param), hint="a short name needs its alias pack, or write the full URI")

    def compares(self, uri, facts):
        """Whether a definition takes predictions and a target, so that its target selector has to resolve."""
        kind = kalfa_kind(uri)
        if kind == "criterion":
            return True
        return kind == "metric" and (not facts.uses or "predictions" in facts.uses)

    def target_fields(self):
        """The target fields in the order the plan builds them: pattern by pattern, column by column."""
        if self.header is None:
            return None
        data = self.data.get("data") or {}
        drop = data.get("drop") or []
        columns = [name for name in self.header["columns"] if name not in drop]
        fields = data.get("fields") or {}
        if not isinstance(fields, dict):
            return None
        owners, _ = assign_fields(columns, list(fields))
        found = []
        for pattern, spec in fields.items():
            if (spec or {}).get("target"):
                found.extend([column for column in columns if owners.get(column) == pattern])
        return found

    def output_wires(self):
        """The output wires of the predicts model, None when they cannot be read from the config."""
        training = self.data.get("training") or {}
        name = training.get("predicts")
        if name is None:
            name = self.trained[0] if len(self.trained) == 1 else None
        if isinstance(name, str) and name.endswith(".ema"):
            name = name[:-4]
        definition = self.models.get(name) if isinstance(name, str) else None
        outputs = (definition or {}).get("outputs")
        return list(outputs) if isinstance(outputs, list) else None

    def target_selector(self, entry):
        """The target a definition compares against: what it writes, else the training.targets entry of its wire."""
        if entry.get("target") is not None:
            return entry["target"]
        targets = (self.data.get("training") or {}).get("targets")
        if not isinstance(targets, dict) or not targets:
            return None
        if entry.get("output") is not None:
            return targets.get(entry["output"])
        return next(iter(targets.values())) if len(targets) == 1 else None

    def definition_target(self, section, name, entry, path):
        selector = self.target_selector(entry)
        fields = self.target_fields()
        if selector == "input" or not fields:
            return
        if selector is None:
            if len(fields) > 1:
                self.error("target_missing", f"{section}.{name} compares against a target and the data has "
                                             f"{len(fields)} target fields; write target on the definition, or "
                                             f"training.targets for the output wire it names", path)
            return
        self.selector_fields(selector, fields, f"{section}.{name}.target", path)

    def selector_fields(self, selector, fields, label, path):
        """A target selector against the target fields: a glob has to match, a name may still be a feed key."""
        matched = expand_targets(selector, fields)
        unknown = [field for field in matched if field not in fields]
        if not matched:
            self.error("target_not_a_field", f"{label} names {selector!r}, which matches no target field; the "
                                             f"target fields are {fields}", path)
        elif unknown:
            self.warning("target_not_a_field", f"{label} names {unknown}, which the data does not carry as target "
                                               f"fields; only a feed that writes them puts them in the batch", path)

    def targets_of(self, training):
        """The shape of training.targets: a mapping of output wire to selector (the fields need the header)."""
        targets = training.get("targets")
        if targets is None:
            return
        if not isinstance(targets, dict):
            self.error("invalid_value", "training.targets must be a mapping of output wire to target fields",
                       ("training", "targets"))
            return
        wires = self.output_wires()
        for wire, selector in targets.items():
            path = ("training", "targets", wire)
            if wires is not None and wire not in wires:
                self.error("targets_not_a_wire", f"training.targets names {wire!r}, which is no output wire of the "
                                                 f"predicts model; the wires are {wires}", path)
            if not is_selector(selector):
                self.error("invalid_value", f"training.targets.{wire} must be a field name, a list of names or a "
                                            f"glob", path)

    def target_fields_of(self, training):
        """training.targets against the target fields of the data, once the header is read."""
        fields = self.target_fields()
        if not fields:
            return
        targets = training.get("targets")
        if targets is None:
            wires = self.output_wires()
            if wires is not None and len(fields) > 1 and len(wires) > 1:
                self.error("targets_missing", f"the predicts model writes {len(wires)} output wires {wires} and the "
                                              f"data has {len(fields)} target fields; write training.targets to say "
                                              f"which wire predicts which fields", ("training",),
                           hint="without it nothing pairs the outputs with the targets and the report writes no "
                                "predictions")
            return
        if not isinstance(targets, dict):
            return
        for wire, selector in targets.items():
            if is_selector(selector):
                self.selector_fields(selector, fields, f"training.targets.{wire}", ("training", "targets", wire))

    def uses_predicts(self):
        for table in (self.losses, self.metrics):
            for entry in table.values():
                uri = entry.get("uri") if isinstance(entry, dict) else entry
                if not isinstance(uri, str):
                    continue
                facts = self.registry.facts(uri)
                if kalfa_kind(uri) == "criterion":
                    return True
                if kalfa_kind(uri) == "metric" and (not facts.uses or "predictions" in facts.uses):
                    return True
        return False

    def predicts_of(self, training):
        predicts = training.get("predicts")
        if predicts is not None:
            base = predicts[:-4] if isinstance(predicts, str) and predicts.endswith(".ema") else predicts
            known = base in self.models
            if isinstance(predicts, str) and predicts.endswith(".ema"):
                known = known and isinstance(self.models.get(base, {}).get("ema"), dict)
            if not known:
                self.error("unresolved_ref", f"training.predicts names {predicts!r}, which is no model",
                           ("training", "predicts"))
            return
        if len(self.trained) != 1 and self.uses_predicts():
            self.error("predicts_required", "losses or metrics need the predicts model and there is not exactly "
                                            "one trained model; write training.predicts", ("training",))

    def order_of(self, training, turn_uri):
        turn = training.get("turn")
        raw_turn = (self.raw.get("training") or {}).get("turn")
        params = turn.get("params") if isinstance(turn, dict) else {}
        params = params or {}
        if raw_turn == "supervised" and len(self.optimizers) > 1:
            self.error("turn_order", "supervised is the single optimizer turn; write alternating with order",
                       ("training", "turn"))
        order = params.get("order")
        if order is not None:
            if not isinstance(order, list):
                self.error("turn_order", "order must be a list of optimizer names", ("training", "turn"))
            else:
                for name in order:
                    if name not in self.optimizers:
                        self.error("turn_order", f"order names {name!r}, which is no optimizer", ("training", "turn"))
        for name in (params.get("steps") or {}) if isinstance(params.get("steps"), dict) else []:
            if name not in self.optimizers:
                self.error("turn_order", f"steps names {name!r}, which is no optimizer", ("training", "turn"))

    def plots_section(self):
        plots = self.data.get("plots") or {}
        if not isinstance(plots, dict):
            self.error("invalid_section", "plots must be a mapping", ("plots",))
            return
        nameless = {}
        for name, entry in plots.items():
            path = ("plots", name)
            if isinstance(entry, dict) and not self.keys(entry, PLOT_KEYS, path, ("uri",)):
                continue
            uri = self.call_of(entry, path, ("plot",), f"plots.{name}")
            if isinstance(entry, dict):
                self.refs_of(uri, entry.get("params"), path)
                self.plot_inputs_of(uri, entry.get("inputs"), path)
            if not isinstance(uri, str):
                continue
            names = self.parameters(self.registry.resolve_quietly(uri))
            if names is not None and "name" not in names:
                if uri in nameless:
                    self.error("plot_name_clash", f"plots.{name} and plots.{nameless[uri]} both use {uri}, which "
                                                  f"takes no name and writes one file", path,
                               hint="a plot lego that takes name writes plots/<definition>.<ext>; one without it "
                                    "can be used once")
                else:
                    nameless[uri] = name

    def sweep_section(self):
        from .std.strategy import Choices, grid_values, parse_space
        from .sweep import OBJECTIVE_KEYS, SWEEP_KEYS

        section = self.data.get("sweep")
        if section is None:
            return
        if not self.keys(section, SWEEP_KEYS, ("sweep",), SWEEP_KEYS):
            return
        uri = self.call_of(section["strategy"], ("sweep", "strategy"), ("strategy",), "sweep.strategy")
        try:
            space = parse_space(section.get("space"))
        except ValueError as exc:
            self.error("sweep_space", str(exc), ("sweep", "space"))
            space = {}
        params = self.data.get("params") or {}
        for name, entry in space.items():
            if name not in params:
                self.error("unresolved_ref", f"sweep.space names {name!r}, which params does not define",
                           ("sweep", "space", name), hint="every swept name is a params entry the config reads as $name$")
            if uri == "/strategy/kalfa/grid" and not isinstance(entry, Choices):
                try:
                    grid_values(name, entry)
                except ValueError as exc:
                    self.error("sweep_space", str(exc), ("sweep", "space", name))
        objective = section.get("objective")
        if not self.keys(objective, OBJECTIVE_KEYS, ("sweep", "objective"), ("monitor",)):
            return
        self.monitors_of({"params": {"monitor": objective.get("monitor")}}, ("sweep", "objective"), strict=False)
        if objective.get("mode", "min") not in ("min", "max"):
            self.error("invalid_value", "sweep.objective.mode is min or max", ("sweep", "objective", "mode"))
        if objective.get("at", "best") not in ("best", "last"):
            self.error("invalid_value", "sweep.objective.at is best or last", ("sweep", "objective", "at"))
        if not isinstance(section.get("record"), str):
            self.error("invalid_value", "sweep.record is the root directory of the points", ("sweep", "record"))

    def plot_inputs_of(self, uri, inputs, path):
        if inputs is None:
            return
        if not isinstance(inputs, dict) or not all(isinstance(value, str) for value in inputs.values()):
            self.error("invalid_value", "plots inputs map a parameter to a history key, a field or a model name", path)
            return
        if not isinstance(uri, str):
            return
        refs = self.registry.facts(uri).refs
        names = self.parameters(self.registry.resolve_quietly(uri))
        for param, name in inputs.items():
            if names is not None and param not in names:
                self.error("signature_mismatch", f"{uri} has no parameter {param!r} to bind an input to",
                           path + ("inputs", param))
            elif param not in refs:
                self.error("invalid_value", f"{uri} declares no refs type for {param!r}; inputs need one "
                                            f"(history, field or model)", path + ("inputs", param))
            elif refs[param] == "model" and name not in self.models:
                self.error("unresolved_ref", f"plots input {param}: {name!r} is no model", path + ("inputs", param))
            elif refs[param] == "history":
                prefix, _, base = name.partition("/")
                if prefix not in HISTORY_SETS or base.split("/")[0] not in {**self.losses, **self.metrics}:
                    self.error("unresolved_ref", f"plots input {param}: {name!r} is no history key",
                               path + ("inputs", param))

    def generate_section(self):
        generate = self.data.get("generate")
        if generate is not None:
            uri = self.call_of(generate, ("generate",), ("generate",), "generate")
            if isinstance(generate, dict):
                self.refs_of(uri, generate.get("params"), ("generate",))

    def source_header(self):
        data = self.data.get("data") or {}
        source = data.get("source")
        if not isinstance(source, dict):
            return None
        uri = source.get("uri")
        params = source.get("params") or {}
        path = params.get("path")
        if not isinstance(uri, str) or not isinstance(path, str):
            return None
        if not Path(path).exists():
            self.error("source_missing", f"data source {path!r} does not exist", ("data", "source"))
            return None
        fields = data.get("fields") or {}
        if uri == "/source/kalfa/image_folder" and isinstance(fields, dict):
            for name in fields:
                if any(char in str(name) for char in "*?["):
                    self.error("invalid_value", f"field {name!r}: a Dataset source takes field names, not globs",
                               ("data", "fields", name))
        try:
            return read_header(uri, params)
        except Exception as exc:
            self.error("source_unreadable", f"cannot read the header of {path!r}: {exc}", ("data", "source"))
            return None

    def data_header(self):
        data = self.data.get("data") or {}
        header = self.source_header()
        if header is None:
            return
        self.header = header
        drop = data.get("drop") or []
        for column in drop:
            if column not in header["columns"]:
                self.warning("drop_missing", f"drop names {column!r}, which is no column", ("data", "drop"))
        columns = [name for name in header["columns"] if name not in drop]
        fields = data.get("fields") or {}
        if not isinstance(fields, dict):
            return
        owners, problems = assign_fields(columns, list(fields))
        for kind, message in problems:
            self.error(kind, message, ("data", "fields"))
        for column, path in self.column_refs():
            if column not in header["columns"]:
                self.error("unresolved_ref", f"{column!r} is no column of the data", path)
            elif column in drop:
                self.error("dropped_column_ref", f"column {column!r} is referenced by a lego and cannot be dropped",
                           path)
            elif column in owners:
                self.error("column_in_fields", f"column {column!r} is referenced by a lego; it is not a field and "
                                               f"does not enter the model", path)
        for column, pattern in owners.items():
            spec = fields.get(pattern) or {}
            chain = spec.get("preprocessors") or []
            if not chain and torch_dtype(header["dtypes"].get(column)) is None:
                self.error("dtype_unsupported", f"column {column!r} has dtype {header['dtypes'].get(column)}; "
                                                f"it needs a preprocessor (cast, one_hot, label_encoder)",
                           ("data", "fields", pattern))
            if column == RESERVED_INPUT:
                self.error("reserved_field", f"column {column!r} is a reserved field name", ("data", "fields"))
            if column == RESERVED_FEATURES and not spec.get("target"):
                self.error("reserved_field", f"column {RESERVED_FEATURES!r} is reserved for the feature tensor",
                           ("data", "fields"))

    def sizes(self):
        header = self.header if self.header is not None else self.source_header()
        if header is None:
            return None
        data = self.data.get("data") or {}
        split = data.get("split")
        params = split.get("params") if isinstance(split, dict) and "uri" in split else split
        ratios = (params or {}).get("ratios") if isinstance(params, dict) else None
        if isinstance(ratios, list) and (not isinstance(split, dict) or "uri" not in split
                                         or split.get("uri") in RATIO_SPLITS):
            try:
                return split_sizes(header["rows"], ratios)
            except ValueError:
                return None
        if isinstance(split, dict) and split.get("uri") == "/split/kalfa/kfold" and isinstance(params, dict):
            try:
                from .std.split import kfold_sizes

                return kfold_sizes(header["rows"], params)
            except (ValueError, TypeError, KeyError):
                return None
        if isinstance(split, dict) and split.get("uri") == "/split/kalfa/given" and isinstance(params, dict):
            return self.given_sizes(header, params)
        return {"train": None, "valid": None, "test": None}

    def given_sizes(self, header, params):
        from .std.source import header as source_header

        found = {"train": header["rows"]}
        source = (self.data.get("data") or {}).get("source") or {}
        for name in ("valid", "test"):
            path = params.get(name)
            if path is None:
                found[name] = 0
                continue
            try:
                other = source_header(source.get("uri"), {**(source.get("params") or {}), "path": path})
            except Exception:
                other = None
            found[name] = other["rows"] if other else None
        return found


def is_selector(value):
    """A target selector: one field name, a glob, or a list of names."""
    return isinstance(value, str) or (isinstance(value, list) and all(isinstance(item, str) for item in value))


def turn_extras(registry, uri, target=None):
    """The training keys a turn lego accepts: its extras fact."""
    return set(registry.facts(uri).extras)


def _mentions(value, uri):
    if isinstance(value, dict):
        return value.get("uri") == uri or any(_mentions(item, uri) for item in value.values())
    if isinstance(value, list):
        return any(_mentions(item, uri) for item in value)
    return False


def sets_text(sizes, loaded=False):
    label = "sets (loaded)" if loaded else "sets (before filters, from the file header)"
    if sizes is None:
        return f"{label}: unknown (the data header could not be read)"
    parts = []
    for name in SETS:
        size = sizes.get(name)
        parts.append(f"{name} {'?' if size is None else size if size else 'none'}")
    return f"{label}: " + ", ".join(parts)


def is_nan(value):
    return isinstance(value, float) and math.isnan(value)
