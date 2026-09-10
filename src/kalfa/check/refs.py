from ..config import resolve_alias
from ..kinds import kalfa_kind, names_of
from ..std.common.optional import installed
from ..std.common.runtime import expand_targets
from ..std.pre.base import assign_fields


def relative_effect(value):
    if len(value) != 1 or next(iter(value)) not in ("times", "plus"):
        return False
    amount = next(iter(value.values()))
    return isinstance(amount, (int, float)) and not isinstance(amount, bool)


class RefRules:
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
            if isinstance(value, dict) and not relative_effect(value):
                self.error("set_value", f"set {key}: a relative effect is {{times: x}} or {{plus: x}} with a number",
                           path)
            return
        if isinstance(value, dict):
            self.error("set_value", f"set {key}: a relative effect ({{times}}, {{plus}}) changes an optimizer "
                                    f"param only", path)
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
                if ref_type == "loss" and value not in self.losses:
                    self.error("set_value", f"set {key} names {value!r}, which losses does not define", path)
                elif ref_type != "loss" and resolve_alias(value, self.surface.aliases) is None:
                    self.error("set_value", f"set {key} names {value!r}, which is no known {ref_type}", path)
            return
        self.error("set_target", f"set target {key!r} names neither an optimizer, a loss nor a trained model", path)

    def refs_of(self, uri, params, path):
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
                    self.error("unresolved_ref", f"{param}: {value!r} is no losses definition",
                               path + ("params", param))
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
        kind = kalfa_kind(uri)
        if kind == "criterion":
            return True
        uses = names_of(facts.get("uses"))
        return kind == "metric" and (not uses or "predictions" in uses)

    def target_fields(self):
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
        matched = expand_targets(selector, fields)
        unknown = [field for field in matched if field not in fields]
        if not matched:
            self.error("target_not_a_field", f"{label} names {selector!r}, which matches no target field; the "
                                             f"target fields are {fields}", path)
        elif unknown:
            self.warning("target_not_a_field", f"{label} names {unknown}, which the data does not carry as target "
                                               f"fields; only a feed that writes them puts them in the batch", path)

    def requires_of(self, uri, name, path):
        for library in names_of(self.registry.facts(uri).get("requires")):
            if not installed(library):
                self.warning("library_missing", f"plots.{name} ({uri}) wants {library}, which is not installed; "
                                                f"the plot is skipped or drawn without it", path,
                             hint=f"pip install {library}")
