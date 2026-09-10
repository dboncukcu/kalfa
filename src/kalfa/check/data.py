from pathlib import Path

import pandas

from ..std.common.runtime import parameter_names
from ..std.pre.base import assign_fields, torch_dtype


def empty_table(header):
    columns = {}
    for name in header["columns"]:
        try:
            columns[name] = pandas.Series(dtype=header["dtypes"].get(name, "object"))
        except TypeError:
            columns[name] = pandas.Series(dtype=object)
    return pandas.DataFrame(columns)


class DataRules:
    def train_steps(self, data):
        steps = []
        for item in data.get("transform") or []:
            if isinstance(item, str):
                steps.append({"uri": self.contract.wiring["filter"], "params": {"query": item}})
            elif isinstance(item, dict) and isinstance(item.get("uri"), str) \
                    and (not item.get("sets") or "train" in item["sets"]):
                steps.append(item)
        return steps

    def foreseen_header(self, header):
        data = self.data.get("data") or {}
        steps = self.train_steps(data)
        frames = [item for item in data.get("frame") or []
                  if isinstance(item, dict) and isinstance(item.get("uri"), str)]
        if header is None or not (steps or frames):
            return header
        table = empty_table(header)
        try:
            for call in steps:
                table = self.registry.resolve(call["uri"])(table, **(call.get("params") or {}))
            for call in frames:
                built = self.registry.resolve(call["uri"])(**(call.get("params") or {}))
                built.fit(table)
                table = built.apply(table)
        except Exception as exception:
            self.warning("columns_unforeseen", f"the columns after the transforms cannot be foreseen on an empty "
                                               f"table: {type(exception).__name__}: {exception}",
                         ("data", "transform"))
            return header
        return {"columns": list(table.columns), "dtypes": {name: str(table[name].dtype) for name in table.columns},
                "rows": header["rows"]}

    def grouped_order(self, data):
        grouped = {name for name, entry in self.preprocessors.items()
                   if isinstance(entry, dict) and isinstance(entry.get("uri"), str)
                   and self.registry.facts(entry["uri"]).get("grouped", False)}
        seen = {}
        fields = data.get("fields") or {}
        if not isinstance(fields, dict):
            return
        for pattern, spec in fields.items():
            chain = [pre for pre in ((spec or {}).get("preprocessors") or []) if pre in grouped]
            for position, name in enumerate(chain):
                for later in chain[position + 1:]:
                    if (later, name) in seen:
                        self.error("grouped_order", f"preprocessors {name!r} and {later!r} both fit over all their "
                                                    f"columns and are written in both orders "
                                                    f"({seen[(later, name)]!r} and {pattern!r}); one order for all "
                                                    f"the fields, or a second definition under another name",
                                   ("data", "fields", pattern))
                    seen[(name, later)] = pattern

    def lazy_rules(self, data):
        hint = "the lazy set streams the table: split with sequential or given, feed with table, no balanced " \
               "sampler and no class_weights"
        split = data.get("split")
        if isinstance(split, dict) and "uri" not in split:
            split = self.contract.wiring["default_split"]
        if self.fact_of(split, "needs_table"):
            self.error("lazy_split", "a stream source cannot be shuffled or folded", ("data", "split"), hint=hint)
        batch = data.get("batch")
        if isinstance(batch, dict) and batch.get("balanced"):
            self.error("lazy_batch", "a stream source cannot be counted for the balanced sampler",
                       ("data", "batch"), hint=hint)
        if self.fact_of(data.get("feed"), "needs_table"):
            self.error("lazy_feed", "the feed needs the table in memory", ("data", "feed"), hint=hint)
        if data.get("frame"):
            self.error("lazy_frame", "a stream source cannot fit a frame transform; it needs the table in memory",
                       ("data", "frame"), hint=hint)
        if data.get("mask") is not None:
            self.error("lazy_mask", "a stream source cannot carry a mask; it needs the table in memory",
                       ("data", "mask"), hint=hint)
        for position, item in enumerate(data.get("transform") or []):
            if isinstance(item, dict) and self.fact_of(item, "needs_table"):
                self.error("lazy_transform", f"transform {position} ({item.get('uri')}) needs the table in memory; "
                                             f"a stream takes filter only", ("data", "transform", position), hint=hint)
        for name, entry in (self.data.get("losses") or {}).items():
            for uri in uris_in(entry):
                if self.registry.lookup(uri) is not None and self.registry.facts(uri).get("counts"):
                    self.error("lazy_data", f"losses.{name} builds {uri}, which counts the train set",
                               ("losses", name), hint=hint)
        for name, entry in (data.get("preprocessors") or {}).items():
            uri = entry.get("uri") if isinstance(entry, dict) else entry
            target = self.registry.resolve_quietly(uri) if isinstance(uri, str) else None
            if target is None:
                continue
            try:
                built = target(**((entry.get("params") if isinstance(entry, dict) else None) or {}))
            except Exception:
                continue
            if built.fits and not built.incremental:
                self.warning("lazy_fit", f"preprocessor {name!r} ({uri}) has no partial_fit; on a stream its column "
                                         f"is collected in memory to fit", ("data", "preprocessors", name),
                             hint="scalers fit incrementally; one_hot and label_encoder collect the column")

    def sizes_helper(self, uri):
        helper = self.fact_of(uri, "sizes")
        if not isinstance(helper, str) or self.registry.lookup(helper) is None:
            return None
        return self.registry.resolve(helper)

    def split_sizes(self, uri, params, rows):
        helper = self.sizes_helper(uri)
        if helper is None:
            return None
        arguments = dict(params)
        if "header" in parameter_names(helper):
            arguments["header"] = self.header_reader()
        try:
            return helper(rows, **arguments)
        except (ValueError, TypeError, KeyError):
            return None

    def presence_of(self, uri, params):
        found = self.split_sizes(uri, params, None) or {}
        return {name: None if size is None else size != 0 for name, size in found.items()}

    def column_refs(self):
        data = self.data.get("data") or {}
        found = []
        calls = [("split", data.get("split")), ("feed", data.get("feed"))]
        for name, entry in (data.get("preprocessors") or {}).items():
            calls.append((f"preprocessors.{name}", entry))
        for position, entry in enumerate(data.get("frame") or []):
            calls.append((f"frame.{position}", entry))
        for label, call in calls:
            if not isinstance(call, dict) or not isinstance(call.get("uri"), str):
                continue
            refs = self.registry.facts(call["uri"]).refs
            for param, ref_type in refs.items():
                value = (call.get("params") or {}).get(param)
                if ref_type == "column" and isinstance(value, str):
                    found.append((value, ("data", *label.split("."), "params", param)))
        return found

    def header_reader(self):
        source = (self.data.get("data") or {}).get("source")
        helper = self.fact_of(source, "header")
        if not isinstance(helper, str) or self.registry.lookup(helper) is None:
            return None
        target = self.registry.resolve(helper)
        accepted = parameter_names(target)
        params = {key: value for key, value in (source.get("params") or {}).items() if key in accepted}

        def read(path):
            return target(**{**params, "path": path})

        return read

    def source_header(self):
        data = self.data.get("data") or {}
        source = data.get("source")
        if self.given_header is not None:
            return dict(self.given_header)
        if not isinstance(source, dict):
            return None
        params = source.get("params") or {}
        path = params.get("path")
        if not isinstance(path, str):
            return None
        if not Path(path).exists():
            self.error("source_missing", f"data source {path!r} does not exist", ("data", "source"))
            return None
        fields = data.get("fields") or {}
        if self.fact_of(source, "samples") and isinstance(fields, dict):
            for name in fields:
                if any(char in str(name) for char in "*?["):
                    self.error("invalid_value", f"field {name!r}: a Dataset source takes field names, not globs",
                               ("data", "fields", name))
        read = self.header_reader()
        if read is None:
            return None
        try:
            return read(path)
        except Exception as exception:
            self.error("source_unreadable", f"cannot read the header of {path!r}: {exception}", ("data", "source"))
            return None

    def data_header(self):
        data = self.data.get("data") or {}
        header = self.foreseen_header(self.source_header())
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
            if column == "input":
                self.error("reserved_field", f"column {column!r} is a reserved field name", ("data", "fields"))
            if column == "x" and not spec.get("target"):
                self.error("reserved_field", "column 'x' is reserved for the feature tensor",
                           ("data", "fields"))

    def sizes(self):
        header = self.header if self.header is not None else self.source_header()
        if header is None:
            return None
        split = (self.data.get("data") or {}).get("split")
        if not isinstance(split, dict):
            return None
        if "uri" in split:
            uri, params = split.get("uri"), split.get("params") or {}
        else:
            uri, params = self.contract.wiring["default_split"], split
        found = self.split_sizes(uri, params, header["rows"])
        if found is None:
            return None if "ratios" in params else {name: None for name in self.sets}
        return found



def uris_in(value):
    if isinstance(value, dict):
        found = [value["uri"]] if isinstance(value.get("uri"), str) else []
        for item in value.values():
            found.extend(uris_in(item))
        return found
    if isinstance(value, list):
        return [uri for item in value for uri in uris_in(item)]
    return []
