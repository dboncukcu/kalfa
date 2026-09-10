import inspect
from pathlib import Path

from ..driver import is_composite, is_shortcut, models_of
from ..kinds import kalfa_kind, names_of
from ..record import read_resolved
from ..schema import Schema
from ..std.builder.base import weights_path
from ..std.strategy.base import Choices, grid_values, parse_space


class SectionRules:
    def top(self):
        if not self.keys(self.data, Schema.sections, (), Schema.required):
            return False
        if self.data.get("seed") is None:
            self.warning("no_seed", "seed is not written; torch runs unseeded and two runs differ", ("seed",))
        if self.data.get("device") is not None:
            self.call_of(self.data["device"], ("device",), ("device",), "device")
        if self.data.get("rng") is not None:
            self.call_of(self.data["rng"], ("rng",), ("rng",), "rng")
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
        if not self.keys(data, Schema.data, ("data",), Schema.data_required):
            return
        if "source" in data:
            self.call_of(data["source"], ("data", "source"), ("source",), "data.source")
        for position, item in enumerate(data.get("transform") or []):
            path = ("data", "transform", position)
            if isinstance(item, dict):
                if self.keys(item, Schema.transform, path, ("uri",)):
                    uri = self.call_of(item, path, ("transform",), f"data.transform[{position}]")
                    if uri is not None:
                        self.refs_of(uri, item.get("params"), path)
                    self.sets_of(item.get("sets"), path)
            elif not isinstance(item, str):
                self.error("invalid_value", "a transform is a query string or a lego call {uri, params, sets}", path)
        if data.get("mask") is not None and not isinstance(data["mask"], str):
            self.error("invalid_value", "data.mask is a pandas query string", ("data", "mask"))
        if data.get("mask") is not None and self.fact_of(data.get("source"), "samples"):
            self.error("mask_needs_table", "a Dataset source has no frame to mask; data.mask needs a table",
                       ("data", "mask"))
        for position, item in enumerate(data.get("frame") or []):
            path = ("data", "frame", position)
            uri = self.call_of(item, path, ("frame",), f"data.frame[{position}]")
            if uri is not None and isinstance(item, dict):
                self.refs_of(uri, item.get("params"), path)
        if data.get("frame") and self.fact_of(data.get("source"), "samples"):
            self.error("frame_needs_table", "a Dataset source has no frame to transform; data.frame needs a table",
                       ("data", "frame"))
        split = data.get("split")
        if isinstance(split, dict) and "uri" in split:
            uri = self.call_of(split, ("data", "split"), ("split",), "data.split")
            self.present.update(self.presence_of(uri, split.get("params") or {}))
        elif isinstance(split, dict):
            self.keys(split, Schema.ratio_split, ("data", "split"), ("ratios",))
            ratios = split.get("ratios")
            if not (isinstance(ratios, list) and len(ratios) == 3
                    and all(isinstance(part, (int, float)) and not isinstance(part, bool) for part in ratios)):
                self.error("invalid_value", "split.ratios must be three numbers (train, valid, test)",
                           ("data", "split"))
            else:
                self.present = {name: ratio > 0 for name, ratio in zip(self.sets, ratios)}
                if not self.present["train"]:
                    self.error("invalid_value", "the train ratio must be positive", ("data", "split"))
        elif "split" in data:
            self.error("invalid_call", "data.split must be {ratios, seed} or {uri, params}", ("data", "split"))
        batch = data.get("batch")
        if isinstance(batch, dict):
            self.batch_keys(batch)
        elif batch is not None and (not isinstance(batch, int) or isinstance(batch, bool) or batch <= 0):
            self.error("invalid_value", "data.batch must be a positive size or a mapping with size",
                       ("data", "batch"))
        preprocessors = data.get("preprocessors") or {}
        if isinstance(preprocessors, dict):
            for name, entry in preprocessors.items():
                path = ("data", "preprocessors", name)
                if isinstance(entry, dict):
                    self.keys(entry, Schema.preprocessor, path, ("uri",))
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
                if not self.keys(spec, Schema.field, path):
                    continue
                chain = spec.get("preprocessors") or []
                if not isinstance(chain, list):
                    self.error("invalid_value", f"fields.{name}.preprocessors must be a list", path)
                    chain = []
                for preprocessor in chain:
                    if preprocessor not in self.preprocessors:
                        self.error("unresolved_ref", f"field {name!r} names preprocessor {preprocessor!r}, "
                                                     f"which data.preprocessors does not define", path)
                    used.add(preprocessor)
                if "target" in spec and not isinstance(spec["target"], bool):
                    self.error("invalid_value", f"fields.{name}.target must be a boolean", path)
                if name == "input":
                    self.error("reserved_field", "'input' is a reserved field name", path)
                if name == "x" and not spec.get("target"):
                    self.error("reserved_field", "'x' is reserved for the feature tensor", path)
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
        self.grouped_order(data)
        if "feed" in data:
            self.call_of(data["feed"], ("data", "feed"), ("feed",), "data.feed")
        source = data.get("source")
        if self.fact_of(source, "stream"):
            self.lazy_rules(data)

    def sets_of(self, sets, path):
        if sets is None:
            return
        if not isinstance(sets, list) or any(item not in self.sets for item in sets):
            self.error("invalid_value", f"sets must list some of {self.sets}", path)

    def model_section(self):
        section = self.data["model"]
        if is_shortcut(section) and any(key in section for key in Schema.model):
            self.error("ambiguous_model", "model mixes the single model shortcut with templates or models",
                       ("model",))
            return
        if is_shortcut(section):
            self.keys(section, Schema.definition_of_model, ("model",), Schema.model_required)
            base = ("model",)
        else:
            self.keys(section, Schema.model, ("model",), ("models",))
            base = None
        self.templates, self.models = models_of(section)
        for name, template in self.templates.items():
            path = ("model", "templates", name)
            if name in self.contract.blocks:
                self.error("model_name_reserved", f"template name {name!r} is reserved for a template block", path)
            if self.keys(template, Schema.template, path, ("nodes",)):
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
            if name in self.contract.blocks:
                self.error("model_name_reserved", f"model name {name!r} is reserved for a template block", path,
                           hint=f"reserved names: {list(self.contract.blocks)}")
            if name in self.templates:
                self.error("model_name_reserved", f"model {name!r} has the name of a template", path)
            if base is None and not self.keys(definition, Schema.definition_of_model, path, Schema.model_required):
                continue
            composite = is_composite(definition)
            if composite:
                self.composites.append(name)
                for key in Schema.composite_forbidden:
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
        except Exception as exception:
            self.error("weights_run_missing", f"weights of model {name!r}: cannot read {resolved}: {exception}", path)
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
            self.keys(item, Schema.node, node_path)
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
        roles = self.contract.roles()
        if not self.keys(init, (*roles, *Schema.init), path):
            return
        for role in roles:
            if role in init:
                self.call_of(init[role], path + (role,), ("init",), f"init.{role}")
        for position, entry in enumerate(init.get("patterns") or []):
            entry_path = path + ("patterns", position)
            if not (isinstance(entry, dict) and "match" in entry):
                self.error("invalid_value", "an init pattern is {match, weights | bias | scale}", entry_path)
                continue
            for role in roles:
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
                if not self.keys(entry, Schema.entry, path, ("uri",)):
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
                if facts.get("needs_grad") and entry.get("sets") != ["train"]:
                    self.error("needs_grad_set", f"{section}.{name} needs gradients; write sets: [train]", path)
                if facts.get("needs_grad") and (self.data.get("training") or {}).get("amp"):
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
            if not self.keys(entry, Schema.optimizer, path, ("uri",)):
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
        extras = {key: value for key, value in training.items() if key not in Schema.training_fixed}
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
        checkpoint = training.get("checkpoint")
        if checkpoint is not None:
            self.call_of(checkpoint, ("training", "checkpoint"), ("checkpoint",), "training.checkpoint")
            self.monitors_of(checkpoint, ("training", "checkpoint"), strict=True)
        written = ["last", *names_of(self.fact_of(checkpoint, "writes"))]
        if "report" in training and report not in written:
            self.error("report_mismatch", f"training.report must be last or a checkpoint the policy writes; "
                                          f"{report!r} is not among {sorted(set(written))}", ("training", "report"),
                       hint="report: best needs checkpoint: {uri: best, params: {monitor: ...}}")
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
        if prefix not in self.history_sets:
            self.error("invalid_value", f"monitor {monitor!r} must start with one of "
                                        f"{', '.join(f'{name}/' for name in self.history_sets)}", path)
            return
        if strict and prefix == "test":
            self.error("test_monitor", f"{monitor!r}: stop and checkpoint cannot watch the test set", path)
        if base not in self.losses and base not in self.metrics:
            self.error("unresolved_ref", f"monitor {monitor!r} names {base!r}, which neither losses nor metrics "
                                         f"define", path)
        set_name = self.history_sets[prefix]
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
            if not isinstance(rule, dict) or not self.keys(rule, Schema.rule, path, ("name", "when", "set")):
                continue
            name = rule.get("name")
            if name in seen:
                self.error("duplicate_name", f"rule {name!r} is defined twice", path)
            if isinstance(name, str) and name.startswith("stop_"):
                self.error("invalid_value", f"rule name {name!r} is reserved for stop triggers", path)
            if "sticky" in rule and not isinstance(rule["sticky"], bool):
                self.error("invalid_value", f"rule {name!r}: sticky must be true or false", path + ("sticky",))
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

    def targets_of(self, training):
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
                uses = names_of(facts.get("uses"))
                if kalfa_kind(uri) == "metric" and (not uses or "predictions" in uses):
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
            if isinstance(entry, dict) and not self.keys(entry, Schema.plot, path, ("uri",)):
                continue
            uri = self.call_of(entry, path, ("plot",), f"plots.{name}")
            if isinstance(entry, dict):
                self.refs_of(uri, entry.get("params"), path)
                self.plot_inputs_of(uri, entry.get("inputs"), path)
            if not isinstance(uri, str):
                continue
            self.requires_of(uri, name, path)
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
        section = self.data.get("sweep")
        if section is None:
            return
        if not self.keys(section, Schema.sweep, ("sweep",), Schema.sweep):
            return
        uri = self.call_of(section["strategy"], ("sweep", "strategy"), ("strategy",), "sweep.strategy")
        try:
            space = parse_space(section.get("space"))
        except ValueError as exception:
            self.error("sweep_space", str(exception), ("sweep", "space"))
            space = {}
        params = self.data.get("params") or {}
        for name, entry in space.items():
            if name not in params:
                self.error("unresolved_ref", f"sweep.space names {name!r}, which params does not define",
                           ("sweep", "space", name),
                           hint="every swept name is a params entry the config reads as $name$")
            if self.fact_of(uri, "enumerates") and not isinstance(entry, Choices):
                try:
                    grid_values(name, entry)
                except ValueError as exception:
                    self.error("sweep_space", str(exception), ("sweep", "space", name))
        objective = section.get("objective")
        if not self.keys(objective, Schema.objective, ("sweep", "objective"), ("monitor",)):
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
                if prefix not in self.history_sets or base.split("/")[0] not in {**self.losses, **self.metrics}:
                    self.error("unresolved_ref", f"plots input {param}: {name!r} is no history key",
                               path + ("inputs", param))

    def wired(self, name, path):
        uri = self.contract.wiring[name]
        if self.registry.lookup(uri) is None:
            self.error("unknown_uri", f"wiring.{name} names {uri}, which is not registered", path)
            return None
        return self.registry.resolve(uri)

    def batch_keys(self, batch):
        target = self.wired("loader", ("data", "batch"))
        if target is None:
            return
        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):
            return
        items = [item for name, item in signature.parameters.items() if name not in ("data", "set")]
        if any(item.kind is item.VAR_KEYWORD for item in items):
            return
        self.keys(batch, [item.name for item in items], ("data", "batch"),
                  [item.name for item in items if item.default is item.empty])

    def figures_section(self):
        figures = self.data.get("figures")
        if figures is None:
            return
        if not isinstance(figures, dict):
            self.error("invalid_section", "figures must be a mapping", ("figures",))
            return
        target = self.wired("figures", ("figures",))
        if target is None:
            return
        allowed = self.parameters(target) or set(figures)
        self.keys(figures, sorted(allowed), ("figures",))
        try:
            target(**{key: value for key, value in figures.items() if key in allowed})
        except (TypeError, ValueError) as exception:
            self.error("invalid_value", str(exception), ("figures",))

    def calibrate_section(self):
        table = self.data.get("calibrate")
        if table is None:
            return
        if not isinstance(table, dict):
            self.error("invalid_section", "calibrate must be a mapping of named lego calls", ("calibrate",))
            return
        for name, entry in table.items():
            path = ("calibrate", name)
            uri = self.call_of(entry, path, ("calibrate",), f"calibrate.{name}")
            if isinstance(entry, dict):
                self.refs_of(uri, entry.get("params"), path)

    def generate_section(self):
        generate = self.data.get("generate")
        if generate is not None:
            uri = self.call_of(generate, ("generate",), ("generate",), "generate")
            if isinstance(generate, dict):
                self.refs_of(uri, generate.get("params"), ("generate",))



def is_selector(value):
    return isinstance(value, str) or (isinstance(value, list) and all(isinstance(item, str) for item in value))


def turn_extras(registry, uri, target=None):
    return set(names_of(registry.facts(uri).get("extras")))
