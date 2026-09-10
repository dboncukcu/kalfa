import copy
import fnmatch
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy

from kalfa.kinds import SETS
from kalfa.std.common.log import logger_for, since
from kalfa.std.common.samples import is_samples


logger = logger_for("data.prep")


class Preprocessor:
    rescales = False
    grouped = False

    def apply(self, values):
        raise NotImplementedError


class Scaler(Preprocessor):
    rescales = True

    def inverse(self, values):
        raise NotImplementedError


class Encoder(Preprocessor):
    def fit(self, values):
        raise NotImplementedError


class Tokenizer(Encoder):
    def encode(self, text):
        raise NotImplementedError

    def decode(self, ids):
        raise NotImplementedError


GLOB_CHARS = "*?["
RESERVED_INPUT = "input"
RESERVED_FEATURES = "x"
FLOAT_KINDS = ("float16", "float32", "float64")
INT_KINDS = ("int8", "int16", "int32", "int64", "uint8")


def is_pattern(key):
    return any(char in key for char in GLOB_CHARS)


def specificity(pattern):
    fixed = sum(1 for char in pattern if char not in GLOB_CHARS)
    return fixed, pattern.count("?")


def assign_fields(columns, fields):
    """Map every column to the field spec that owns it, by the glob specificity rules.

    Returns the owner pattern per column and the problems found: a spec that matches nothing,
    and a column two patterns claim with equal specificity.
    """
    owners = {}
    problems = []
    matches = {}
    for pattern in fields:
        hits = [column for column in columns if fnmatch.fnmatchcase(column, pattern)] \
            if is_pattern(pattern) else [column for column in columns if column == pattern]
        matches[pattern] = hits
        if not hits:
            problems.append(("column_missing", f"field {pattern!r} matches no column; the columns are {list(columns)}"))
    for column in columns:
        claims = [pattern for pattern, hits in matches.items() if column in hits]
        if not claims:
            continue
        literal = [pattern for pattern in claims if not is_pattern(pattern)]
        if literal:
            owners[column] = literal[0]
            continue
        ranked = sorted(claims, key=lambda pattern: specificity(pattern), reverse=True)
        if len(ranked) > 1 and specificity(ranked[0]) == specificity(ranked[1]):
            problems.append(("glob_ambiguous",
                             f"column {column!r} matches {ranked[0]!r} and {ranked[1]!r} with equal specificity"))
            continue
        owners[column] = ranked[0]
    return owners, problems


ARROW_FLOATS = ("double", "float", "halffloat", "decimal")
ARROW_INTS = ("int8", "int16", "int32", "int64", "uint8")


def torch_dtype(dtype):
    """The torch facing dtype of a frame or arrow column, or None when a preprocessor is needed first."""
    name = str(dtype)
    if name in FLOAT_KINDS or name.startswith("Float") or name in ARROW_FLOATS or name.startswith("decimal"):
        return "float32"
    if name in INT_KINDS or name.startswith("Int") or name.startswith("UInt") or name in ARROW_INTS:
        return "int64"
    if name in ("bool", "boolean"):
        return "bool"
    return None


@dataclass
class Field:
    name: str
    chain: list
    target: bool
    columns: list = field(default_factory=list)


class ColumnView:
    """One column of a grouped preprocessor under the per column interface the chains use."""

    def __init__(self, obj, position):
        self.obj = obj
        self.position = position

    def __getattr__(self, name):
        return getattr(self.obj, name)

    def apply(self, values):
        return self.obj.apply(values, columns=[self.position])

    def inverse(self, values):
        return self.obj.inverse(values, columns=[self.position])


@dataclass
class Grouped:
    """A preprocessor fitted once over every column that names it, in fit order."""

    obj: object
    columns: list

    def view(self, name):
        return ColumnView(self.obj, self.columns.index(name))


def is_grouped(obj):
    """Whether a preprocessor fits over all its columns at once (sklearn style) instead of one copy per column."""
    return bool(getattr(obj, "grouped", False))


def fitted_object(fitted, pre, name):
    entry = (fitted or {}).get(pre)
    if isinstance(entry, Grouped):
        return entry.view(name) if name in entry.columns else None
    return (entry or {}).get(name)


@dataclass
class Prep:
    """The fitted preprocessing plan: fields in order, fitted objects per (preprocessor, column), output layout."""

    fields: list
    fitted: dict
    sets: dict
    dtypes: dict
    drop: list

    @property
    def features(self):
        out = []
        for item in self.fields:
            if not item.target:
                out.extend(item.columns)
        return out

    @property
    def targets(self):
        return {item.name: list(item.columns) for item in self.fields if item.target}

    def target_names(self):
        return [item.name for item in self.fields if item.target]

    def object_of(self, pre, name):
        """The fitted preprocessor of one column: the copy of a per column lego, a view of a grouped one."""
        return fitted_object(self.fitted, pre, name)

    def inverse(self, name, values, set_name="test"):
        """Undo a target field's chain on a column of values (numpy, one column)."""
        item = next(entry for entry in self.fields if entry.name == name)
        out = numpy.asarray(values)
        for pre in reversed(item.chain):
            if not self._applies(pre, set_name):
                continue
            obj = self.object_of(pre, name)
            if hasattr(obj, "inverse"):
                out = numpy.asarray(obj.inverse(out))
        return out

    def rescale(self, name, values, set_name="test"):
        """Undo only the rescaling preprocessors of a target field's chain (scalers, log); encoders stay."""
        item = next(entry for entry in self.fields if entry.name == name)
        out = numpy.asarray(values)
        for pre in reversed(item.chain):
            if not self._applies(pre, set_name):
                continue
            obj = self.object_of(pre, name)
            if getattr(obj, "rescales", False) and hasattr(obj, "inverse"):
                out = numpy.asarray(obj.inverse(out))
        return out

    def rescales(self, name=None):
        """Whether the chain of the named field, or of any field, has a rescaling preprocessor."""
        for item in self.fields:
            if name is not None and item.name != name:
                continue
            for pre in item.chain:
                obj = self.object_of(pre, item.name)
                if getattr(obj, "rescales", False) and hasattr(obj, "inverse"):
                    return True
        return False

    def field_of(self, column):
        """The field a produced column belongs to."""
        for item in self.fields:
            if column in item.columns:
                return item.name
        return None

    def rescale_features(self, matrix, set_name="test"):
        """Undo the rescaling preprocessors column by column of a feature matrix laid out as ``features``."""
        out = numpy.array(matrix, dtype="float64")
        if out.ndim != 2 or out.shape[1] != len(self.features):
            return out
        for position, column in enumerate(self.features):
            name = self.field_of(column)
            if name is not None:
                out[:, position] = self.rescale(name, out[:, position], set_name)
        return out

    def tokenizer(self):
        """The fitted preprocessor that encodes and decodes text, if any field has one."""
        for item in self.fields:
            for pre in item.chain:
                obj = self.object_of(pre, item.name)
                if obj is not None and hasattr(obj, "encode") and hasattr(obj, "decode"):
                    return obj
        return None

    def decoder(self, name):
        """The preprocessor of a target field's chain that decodes class scores, if any."""
        item = next(entry for entry in self.fields if entry.name == name)
        for pre in reversed(item.chain):
            obj = self.object_of(pre, name)
            if hasattr(obj, "decode"):
                return obj
        return None

    def decode(self, name, scores, set_name="test"):
        """Class scores of a target field as labels: the decoding preprocessor, then the rest of the chain inverted."""
        item = next(entry for entry in self.fields if entry.name == name)
        out = None
        for pre in reversed(item.chain):
            obj = self.object_of(pre, name)
            if out is None:
                if hasattr(obj, "decode"):
                    out = numpy.asarray(obj.decode(scores))
                continue
            if self._applies(pre, set_name) and hasattr(obj, "inverse"):
                out = numpy.asarray(obj.inverse(out))
        return out

    def _applies(self, pre, set_name):
        allowed = self.sets.get(pre)
        return allowed is None or set_name in allowed

    def plan(self):
        return {"fields": [{"name": item.name, "chain": item.chain, "target": item.target, "columns": item.columns}
                           for item in self.fields],
                "sets": self.sets, "dtypes": self.dtypes, "drop": self.drop}


@dataclass
class Frame:
    """One set after preprocessing.

    Table mode: ``data`` holds the typed columns, ``features`` the feature order, ``targets`` the target layout and
    ``extra`` the columns that are neither fields nor dropped (they reach legos by name and never the model).
    Dataset mode: ``dataset`` is the sample store, ``fields`` the field names in order, ``chains`` the fitted
    preprocessors per field that apply to this set, applied per item by the feed.
    """

    data: object
    features: list
    targets: dict
    set: str
    extra: object = None
    dataset: object = None
    fields: list = None
    chains: dict = None
    stream: object = None

    @property
    def index(self):
        if self.stream is not None:
            return None
        if self.dataset is not None:
            return self.dataset.index
        return self.data.index

    def __len__(self):
        if self.stream is not None:
            raise TypeError("a stream frame has no length; it is read in chunks")
        if self.dataset is not None:
            return len(self.dataset)
        return len(self.data)


class StreamView:
    """One set of a stream source after preprocessing: chunks of typed columns laid out like the table mode."""

    def __init__(self, stream, prep, set_name, sets):
        self.stream = stream
        self.prep = prep
        self.set_name = set_name
        self.sets = sets

    def chunks(self):
        import pandas

        for chunk in self.stream.chunks():
            columns = {}
            for item in self.prep.fields:
                values = run_chain(item, chunk[item.name].to_numpy(), self.prep.fitted, {}, self.sets, self.set_name,
                                    fit=False)
                if len(values):
                    values, _ = cast_values(values, item.name)
                if values.ndim == 1:
                    columns[item.columns[0]] = values
                else:
                    for position, column in enumerate(item.columns):
                        columns[column] = values[:, position]
            yield numpy.asarray(chunk.index), pandas.DataFrame(columns, index=chunk.index)


def columns_of_source(df):
    if is_samples(df):
        return list(df.fields)
    return list(df.columns)


def fit_stream(items, df, templates, sets, fitted, dtypes):
    """Fit every chain on a stream: one pass per chain position, incremental where the preprocessor has
    partial_fit, on the collected column otherwise; the layout comes from the first chunk."""
    for item in items:
        earlier = []
        for pre in item.chain:
            allowed = sets.get(pre)
            if allowed is not None and "train" not in allowed:
                continue
            obj = copy.deepcopy(templates[pre])
            if hasattr(obj, "fit"):
                collected = []
                for chunk in df.chunks():
                    values = chunk[item.name].to_numpy()
                    for previous in earlier:
                        values = numpy.asarray(previous.apply(values))
                    if hasattr(obj, "partial_fit"):
                        obj.partial_fit(values)
                    else:
                        collected.append(values)
                if collected:
                    obj.fit(numpy.concatenate(collected))
                elif not hasattr(obj, "partial_fit"):
                    obj.fit(numpy.zeros(0))
            fitted.setdefault(pre, {})[item.name] = obj
            earlier.append(obj)
        head = df.head()
        values = head[item.name].to_numpy() if head is not None else numpy.zeros(0)
        values = run_chain(item, values, fitted, templates, sets, "train", fit=False)
        if len(values):
            values, kind = cast_values(values, item.name)
        else:
            kind = torch_dtype(values.dtype) or "float32"
        item.columns = columns_of(item, values, fitted)
        for column in item.columns:
            dtypes[column] = kind


def resolve_fields(df, fields, drop):
    columns = [name for name in columns_of_source(df) if name not in (drop or [])]
    if is_samples(df):
        patterns = [pattern for pattern in fields if is_pattern(pattern)]
        if patterns:
            raise ValueError(f"pattern fields {patterns} need a table; a Dataset source yields the fields it "
                             f"declares, list them by name")
    owners, problems = assign_fields(columns, fields)
    errors = [message for kind, message in problems]
    if errors:
        raise ValueError("; ".join(errors))
    resolved = []
    for pattern, spec in fields.items():
        spec = spec or {}
        chain = list(spec.get("preprocessors") or [])
        target = bool(spec.get("target", False))
        for column in columns:
            if owners.get(column) == pattern:
                if column == RESERVED_INPUT:
                    raise ValueError(f"{RESERVED_INPUT!r} is a reserved field name")
                if column == RESERVED_FEATURES and not target:
                    raise ValueError(f"{RESERVED_FEATURES!r} is reserved for the feature tensor; rename the column")
                resolved.append(Field(column, chain, target))
    return resolved


def fit_chains(items, values_of, templates, sets, fitted, set_name="train"):
    """Fit every chain on the train set: a per column preprocessor gets one fitted copy per column, a grouped one
    is fitted once over the matrix of every column that names it. Returns the transformed values per field."""
    values = {item.name: values_of(item) for item in items}
    position = {item.name: 0 for item in items}

    def skipped(pre):
        allowed = sets.get(pre)
        return allowed is not None and set_name not in allowed

    def waiting(item):
        while position[item.name] < len(item.chain):
            pre = item.chain[position[item.name]]
            if skipped(pre):
                position[item.name] += 1
                continue
            return pre
        return None

    while True:
        for item in items:
            pre = waiting(item)
            while pre is not None and not is_grouped(templates[pre]):
                obj = copy.deepcopy(templates[pre])
                if hasattr(obj, "fit"):
                    obj.fit(values[item.name])
                fitted.setdefault(pre, {})[item.name] = obj
                values[item.name] = numpy.asarray(obj.apply(values[item.name]))
                position[item.name] += 1
                pre = waiting(item)
        ready = None
        for item in items:
            pre = waiting(item)
            if pre is None:
                continue
            members = [other for other in items if pre in other.chain]
            if all(waiting(other) == pre for other in members):
                ready = (pre, members)
                break
        if ready is None:
            break
        pre, members = ready
        wide = [member.name for member in members if numpy.asarray(values[member.name]).ndim > 1]
        if wide:
            raise ValueError(f"preprocessor {pre!r} fits over all its columns at once, so every column reaching it "
                             f"must be one column wide; {wide} are wider")
        obj = copy.deepcopy(templates[pre])
        block = numpy.column_stack([numpy.asarray(values[member.name]) for member in members])
        if hasattr(obj, "fit"):
            obj.fit(block)
        transformed = numpy.asarray(obj.apply(block))
        for index, member in enumerate(members):
            values[member.name] = transformed[:, index]
            position[member.name] += 1
        fitted[pre] = Grouped(obj, [member.name for member in members])
    stuck = [item.name for item in items if waiting(item) is not None]
    if stuck:
        raise ValueError(f"the chains of {stuck} cannot be fitted: preprocessors that fit over all their columns "
                         f"are written in different orders")
    return values


def run_chain(item, values, fitted, templates, sets, set_name, fit):
    for pre in item.chain:
        allowed = sets.get(pre)
        if allowed is not None and set_name not in allowed:
            continue
        if fit:
            obj = copy.deepcopy(templates[pre])
            if hasattr(obj, "fit"):
                obj.fit(values)
            fitted.setdefault(pre, {})[item.name] = obj
        else:
            obj = fitted_object(fitted, pre, item.name)
        values = numpy.asarray(obj.apply(values))
    return values


def columns_of(item, values, fitted):
    if values.ndim == 1:
        return [item.name]
    for pre in reversed(item.chain):
        obj = fitted_object(fitted, pre, item.name)
        if obj is not None and hasattr(obj, "columns"):
            return list(obj.columns(item.name))
    return [f"{item.name}_{position}" for position in range(values.shape[1])]


def cast_values(values, name):
    kind = torch_dtype(values.dtype)
    if kind is None:
        raise TypeError(f"field {name!r} has dtype {values.dtype} after its chain; add a preprocessor that turns it "
                        f"into a number (cast, one_hot, label_encoder, a tokenizer)")
    return values.astype(kind), kind


def values_of(df, name):
    return df[name].to_numpy()


def sets_of(keys):
    """The sets each preprocessor applies to, from the definition keys; a missing entry means every set."""
    found = {}
    for name, entry in (keys or {}).items():
        if isinstance(entry, dict) and entry.get("sets") is not None:
            found[name] = list(entry["sets"])
    return found


def report_fitted(prep, started):
    if logger.isEnabledFor(logging.DEBUG):
        for name, entry in prep.fitted.items():
            columns = entry.columns if isinstance(entry, Grouped) else entry
            logger.debug(f"{name} on {len(columns)} columns")
    logger.info(f"{len(prep.fields)} fields -> {len(prep.features)} features, {len(prep.targets)} targets "
                f"({since(started)})")
    return prep


def write_prep(prep, record):
    target = Path(record) / "preprocessors"
    target.mkdir(parents=True, exist_ok=True)
    for name, per_column in prep.fitted.items():
        with (target / f"{name}.pkl").open("wb") as stream:
            pickle.dump(per_column, stream)
    (target / "plan.json").write_text(json.dumps(prep.plan(), indent=2))


def read_prep(record):
    target = Path(record) / "preprocessors"
    plan = json.loads((target / "plan.json").read_text())
    fitted = {}
    for item in plan["fields"]:
        for pre in item["chain"]:
            if pre in fitted:
                continue
            with (target / f"{pre}.pkl").open("rb") as stream:
                fitted[pre] = pickle.load(stream)
    fields = [Field(item["name"], list(item["chain"]), bool(item["target"]), list(item["columns"]))
              for item in plan["fields"]]
    return Prep(fields, fitted, dict(plan["sets"]), dict(plan["dtypes"]), list(plan["drop"]))


def all_sets():
    return list(SETS)


def pil_image(value):
    from PIL import Image

    if not isinstance(value, Image.Image):
        raise TypeError(f"expected an image, got {type(value).__name__}")
    return value


class ToTensor(Preprocessor):
    """PIL image to a float tensor in [0, 1], channels first (grayscale gets one channel)."""

    dtype = "float32"

    def __init__(self, signed=False):
        self.signed = signed

    def apply(self, value):
        import torch

        if isinstance(value, (tuple, list)):
            return torch.stack([self.apply(item) for item in value])
        array = numpy.asarray(pil_image(value), dtype="float32") / 255.0
        if array.ndim == 2:
            array = array[:, :, None]
        tensor = torch.from_numpy(numpy.ascontiguousarray(array.transpose(2, 0, 1)))
        return tensor * 2.0 - 1.0 if self.signed else tensor
