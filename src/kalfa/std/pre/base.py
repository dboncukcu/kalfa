import copy
import fnmatch
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy
import pandas
import torch

from kalfa.std.common.log import logger_for, since
from kalfa.std.common.samples import is_samples


logger = logger_for("data.prep")


class Preprocessor:
    rescales = False
    grouped = False
    fits = False
    incremental = False
    decodes = False
    dtype = None

    def fit(self, values) -> None:
        raise NotImplementedError

    def apply(self, values):
        raise NotImplementedError

    def inverse(self, values):
        return values

    def columns(self, name: str) -> list | None:
        return None

    def extras(self, values) -> dict:
        return {}


class Scaler(Preprocessor):
    rescales = True

    def inverse(self, values):
        raise NotImplementedError


class Affine(Scaler):
    def affine(self):
        raise NotImplementedError

    def __getstate__(self):
        return {key: value for key, value in self.__dict__.items() if key != "cached_terms"}

    def device_terms(self, tensor, columns=None):
        cache = getattr(self, "cached_terms", None)
        if cache is None:
            cache = self.cached_terms = {}
        key = (str(tensor.device), tensor.dtype, None if columns is None else tuple(columns))
        if key not in cache:
            shift, scale = (numpy.asarray(part, dtype="float64").reshape(-1) for part in self.affine())
            if columns is not None:
                picked = list(columns)
                shift, scale = shift[picked], scale[picked]
            if shift.size == 1:
                cache[key] = (float(shift[0]), float(scale[0]))
            else:
                cache[key] = (torch.as_tensor(shift, dtype=tensor.dtype, device=tensor.device),
                              torch.as_tensor(scale, dtype=tensor.dtype, device=tensor.device))
        return cache[key]

    def inverse_torch(self, tensor, columns=None):
        shift, scale = self.device_terms(tensor, columns)
        if not isinstance(shift, float) and tensor.shape[-1] != shift.numel():
            raise ValueError(f"the scaler was fitted on {shift.numel()} columns and got {tensor.shape[-1]}")
        return tensor * scale + shift


class Encoder(Preprocessor):
    fits = True

    def fit(self, values) -> None:
        raise NotImplementedError


class Tokenizer(Encoder):
    dtype = "int64"

    def encode(self, text: str) -> numpy.ndarray:
        raise NotImplementedError

    def decode(self, ids) -> str:
        raise NotImplementedError

    @property
    def size(self) -> int:
        raise NotImplementedError


def is_pattern(key):
    return any(character in key for character in "*?[")


def matches_any(column, patterns):
    for pattern in patterns or ():
        if fnmatch.fnmatchcase(column, pattern) if is_pattern(pattern) else column == pattern:
            return True
    return False


def specificity(pattern):
    fixed = sum(1 for character in pattern if character not in "*?[")
    return fixed, pattern.count("?")


def assign_fields(columns, fields):
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


def torch_dtype(dtype):
    name = str(dtype)
    if name.startswith(("halffloat", "decimal")) or pandas.api.types.is_float_dtype(name):
        return "float32"
    if pandas.api.types.is_bool_dtype(name):
        return "bool"
    if pandas.api.types.is_integer_dtype(name):
        return "int64"
    return None


@dataclass
class Field:
    name: str
    chain: list
    target: bool
    columns: list = field(default_factory=list)
    extras: list = field(default_factory=list)


class ColumnView:
    def __init__(self, preprocessor, position):
        self.preprocessor = preprocessor
        self.position = position

    def __getattr__(self, name):
        return getattr(self.preprocessor, name)

    def apply(self, values):
        return self.preprocessor.apply(values, columns=[self.position])

    def inverse(self, values):
        return self.preprocessor.inverse(values, columns=[self.position])

    def inverse_torch(self, tensor, columns=None):
        return self.preprocessor.inverse_torch(tensor, columns=[self.position])


@dataclass
class Grouped:
    preprocessor: Preprocessor
    columns: list

    def view(self, name):
        return ColumnView(self.preprocessor, self.columns.index(name))


def fitted_object(fitted, preprocessor, name):
    entry = (fitted or {}).get(preprocessor)
    if isinstance(entry, Grouped):
        return entry.view(name) if name in entry.columns else None
    return (entry or {}).get(name)


def fitted_class(fitted, preprocessor, name):
    entry = (fitted or {}).get(preprocessor)
    if isinstance(entry, Grouped):
        return entry.preprocessor if name in entry.columns else None
    return (entry or {}).get(name)


@dataclass
class Prep:
    fields: list
    fitted: dict
    sets: dict
    dtypes: dict
    drop: list
    spectators: list = field(default_factory=list)

    @property
    def features(self):
        out = []
        for item in self.fields:
            if not item.target:
                out.extend(item.columns)
            out.extend(item.extras)
        return out

    @property
    def targets(self):
        return {item.name: list(item.columns) for item in self.fields if item.target}

    def field(self, name):
        return next(entry for entry in self.fields if entry.name == name)

    def object_of(self, preprocessor, name):
        return fitted_object(self.fitted, preprocessor, name)

    def applies(self, preprocessor, set_name):
        allowed = self.sets.get(preprocessor)
        return allowed is None or set_name in allowed

    def inverse(self, name, values, set_name="test"):
        out = numpy.asarray(values)
        for preprocessor in reversed(self.field(name).chain):
            fitted = self.object_of(preprocessor, name)
            if fitted is not None and self.applies(preprocessor, set_name):
                out = numpy.asarray(fitted.inverse(out))
        return out

    def rescale(self, name, values, set_name="test"):
        out = numpy.asarray(values)
        for preprocessor in reversed(self.field(name).chain):
            if not self.applies(preprocessor, set_name):
                continue
            fitted = self.object_of(preprocessor, name)
            if fitted is not None and fitted.rescales:
                out = numpy.asarray(fitted.inverse(out))
        return out

    def rescale_torch(self, name, tensor, set_name="test"):
        out = tensor
        for preprocessor in reversed(self.field(name).chain):
            if not self.applies(preprocessor, set_name):
                continue
            fitted = self.object_of(preprocessor, name)
            if fitted is not None and fitted.rescales:
                out = fitted.inverse_torch(out)
        return out

    def rescales_on_device(self, names, set_name="test"):
        for name in names:
            for preprocessor in self.field(name).chain:
                if not self.applies(preprocessor, set_name):
                    continue
                built = fitted_class(self.fitted, preprocessor, name)
                if built is not None and built.rescales and not hasattr(built, "inverse_torch"):
                    return False
        return True

    def rescales(self, name=None):
        for item in self.fields:
            if name is not None and item.name != name:
                continue
            for preprocessor in item.chain:
                fitted = self.object_of(preprocessor, item.name)
                if fitted is not None and fitted.rescales:
                    return True
        return False

    def field_of(self, column):
        for item in self.fields:
            if column in item.columns:
                return item.name
        return None

    def rescale_features(self, matrix, set_name="test"):
        out = numpy.array(matrix, dtype="float64")
        if out.ndim != 2 or out.shape[1] != len(self.features):
            return out
        for position, column in enumerate(self.features):
            name = self.field_of(column)
            if name is not None:
                out[:, position] = self.rescale(name, out[:, position], set_name)
        return out

    def tokenizer(self):
        for item in self.fields:
            for preprocessor in item.chain:
                fitted = self.object_of(preprocessor, item.name)
                if isinstance(fitted, Tokenizer):
                    return fitted
        return None

    def decoder(self, name):
        for preprocessor in reversed(self.field(name).chain):
            fitted = self.object_of(preprocessor, name)
            if fitted is not None and fitted.decodes:
                return fitted
        return None

    def decode(self, name, scores, set_name="test"):
        out = None
        for preprocessor in reversed(self.field(name).chain):
            fitted = self.object_of(preprocessor, name)
            if fitted is None:
                continue
            if out is None:
                if fitted.decodes:
                    out = numpy.asarray(fitted.decode(scores))
                continue
            if self.applies(preprocessor, set_name):
                out = numpy.asarray(fitted.inverse(out))
        return out

    def plan(self):
        return {"fields": [{"name": item.name, "chain": item.chain, "target": item.target, "columns": item.columns,
                            "extras": item.extras} for item in self.fields],
                "sets": self.sets, "dtypes": self.dtypes, "drop": self.drop, "spectators": self.spectators}


@dataclass
class Frame:
    features: list
    targets: dict
    set: str

    @property
    def index(self):
        return None

    def __len__(self) -> int:
        raise TypeError(f"a {type(self).__name__} has no length")


@dataclass
class TableFrame(Frame):
    data: object = None
    extra: object = None
    mask: object = None

    @property
    def index(self):
        return self.data.index

    def __len__(self):
        return len(self.data)


@dataclass
class SampleFrame(Frame):
    dataset: object = None
    fields: list = field(default_factory=list)
    chains: dict = field(default_factory=dict)

    @property
    def index(self):
        return self.dataset.index

    def __len__(self):
        return len(self.dataset)


@dataclass
class StreamFrame(Frame):
    stream: object = None

    def __len__(self):
        raise TypeError("a stream frame has no length; it is read in chunks")


class StreamView:
    def __init__(self, stream, prep, set_name, sets):
        self.stream = stream
        self.prep = prep
        self.set_name = set_name
        self.sets = sets

    def chunks(self):
        for chunk in self.stream.chunks():
            columns = {}
            for item in self.prep.fields:
                values, extras = run_chain(item, chunk[item.name].to_numpy(), self.prep.fitted, {}, self.sets,
                                           self.set_name, fit=False)
                if len(values):
                    values, _ = cast_values(values, item.name)
                if values.ndim == 1:
                    columns[item.columns[0]] = values
                else:
                    for position, column in enumerate(item.columns):
                        columns[column] = values[:, position]
                columns.update(typed_extras(extras, self.prep.dtypes))
            yield numpy.asarray(chunk.index), pandas.DataFrame(columns, index=chunk.index)


def columns_of_source(df):
    if is_samples(df):
        return list(df.fields)
    return list(df.columns)


def fit_stream(items, df, templates, sets, fitted, dtypes):
    for item in items:
        earlier = []
        for name in item.chain:
            allowed = sets.get(name)
            if allowed is not None and "train" not in allowed:
                continue
            preprocessor = copy.deepcopy(templates[name])
            if preprocessor.fits:
                fit_on_stream(preprocessor, df, item.name, earlier)
            fitted.setdefault(name, {})[item.name] = preprocessor
            earlier.append(preprocessor)
        head = df.head()
        values = head[item.name].to_numpy() if head is not None else numpy.zeros(0)
        values, extras = run_chain(item, values, fitted, templates, sets, "train", fit=False)
        if len(values):
            values, kind = cast_values(values, item.name)
        else:
            kind = torch_dtype(values.dtype) or "float32"
        item.columns = columns_of(item, values, fitted)
        for column in item.columns:
            dtypes[column] = kind
        item.extras = extra_columns(extras, dtypes)


def fit_on_stream(preprocessor, df, name, earlier):
    collected = []
    for chunk in df.chunks():
        values = chunk[name].to_numpy()
        for previous in earlier:
            values = numpy.asarray(previous.apply(values))
        if preprocessor.incremental:
            preprocessor.partial_fit(values)
        else:
            collected.append(values)
    if collected:
        preprocessor.fit(numpy.concatenate(collected))
    elif not preprocessor.incremental:
        preprocessor.fit(numpy.zeros(0))


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
            if owners.get(column) != pattern:
                continue
            if column == "input":
                raise ValueError("'input' is a reserved field name")
            if column == "x" and not target:
                raise ValueError("'x' is reserved for the feature tensor; rename the column")
            resolved.append(Field(column, chain, target))
    return resolved


class ChainFit:
    def __init__(self, items, values, templates, sets, fitted, set_name):
        self.items = items
        self.values = values
        self.templates = templates
        self.sets = sets
        self.fitted = fitted
        self.set_name = set_name
        self.position = {item.name: 0 for item in items}
        self.sides = {item.name: {} for item in items}

    def skipped(self, name):
        allowed = self.sets.get(name)
        return allowed is not None and self.set_name not in allowed

    def waiting(self, item):
        while self.position[item.name] < len(item.chain):
            name = item.chain[self.position[item.name]]
            if self.skipped(name):
                self.position[item.name] += 1
                continue
            return name
        return None

    def fit_single(self, item, name):
        preprocessor = copy.deepcopy(self.templates[name])
        if preprocessor.fits:
            preprocessor.fit(self.values[item.name])
        self.fitted.setdefault(name, {})[item.name] = preprocessor
        self.sides[item.name].update(named_extras(item.name, preprocessor, self.values[item.name]))
        self.values[item.name] = numpy.asarray(preprocessor.apply(self.values[item.name]))
        self.position[item.name] += 1

    def ready_group(self):
        for item in self.items:
            name = self.waiting(item)
            if name is None:
                continue
            members = [other for other in self.items if name in other.chain]
            if all(self.waiting(other) == name for other in members):
                return name, members
        return None

    def fit_group(self, name, members):
        wide = [member.name for member in members if numpy.asarray(self.values[member.name]).ndim > 1]
        if wide:
            raise ValueError(f"preprocessor {name!r} fits over all its columns at once, so every column reaching it "
                             f"must be one column wide; {wide} are wider")
        preprocessor = copy.deepcopy(self.templates[name])
        block = numpy.column_stack([numpy.asarray(self.values[member.name]) for member in members])
        if preprocessor.fits:
            preprocessor.fit(block)
        transformed = numpy.asarray(preprocessor.apply(block))
        for index, member in enumerate(members):
            self.values[member.name] = transformed[:, index]
            self.position[member.name] += 1
        self.fitted[name] = Grouped(preprocessor, [member.name for member in members])

    def run(self):
        while True:
            for item in self.items:
                name = self.waiting(item)
                while name is not None and not self.templates[name].grouped:
                    self.fit_single(item, name)
                    name = self.waiting(item)
            group = self.ready_group()
            if group is None:
                break
            self.fit_group(*group)
        stuck = [item.name for item in self.items if self.waiting(item) is not None]
        if stuck:
            raise ValueError(f"the chains of {stuck} cannot be fitted: preprocessors that fit over all their columns "
                             f"are written in different orders")
        return self.values


def fit_chains(items, values_of, templates, sets, fitted, set_name="train"):
    values = {item.name: values_of(item) for item in items}
    chains = ChainFit(items, values, templates, sets, fitted, set_name)
    return chains.run(), chains.sides


def named_extras(name, preprocessor, values):
    return {f"{name}_{key}": numpy.asarray(value) for key, value in preprocessor.extras(values).items()}


def extra_columns(extras, dtypes):
    for column, values in extras.items():
        dtypes[column] = torch_dtype(numpy.asarray(values).dtype) or "float32"
    return list(extras)


def typed_extras(extras, dtypes):
    return {column: numpy.asarray(values).astype(dtypes.get(column, numpy.asarray(values).dtype))
            for column, values in extras.items()}


def run_chain(item, values, fitted, templates, sets, set_name, fit):
    extras = {}
    for name in item.chain:
        allowed = sets.get(name)
        if allowed is not None and set_name not in allowed:
            continue
        if fit:
            preprocessor = copy.deepcopy(templates[name])
            if preprocessor.fits:
                preprocessor.fit(values)
            fitted.setdefault(name, {})[item.name] = preprocessor
        else:
            preprocessor = fitted_object(fitted, name, item.name)
        extras.update(named_extras(item.name, preprocessor, values))
        values = numpy.asarray(preprocessor.apply(values))
    return values, extras


def columns_of(item, values, fitted):
    if values.ndim == 1:
        return [item.name]
    for name in reversed(item.chain):
        preprocessor = fitted_object(fitted, name, item.name)
        found = preprocessor.columns(item.name) if preprocessor is not None else None
        if found is not None:
            return list(found)
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
    target = Path(record) / "fitted" / "preprocessors"
    target.mkdir(parents=True, exist_ok=True)
    for name, per_column in prep.fitted.items():
        with (target / f"{name}.pkl").open("wb") as stream:
            pickle.dump(per_column, stream)
    (target / "plan.json").write_text(json.dumps(prep.plan(), indent=2))


def read_prep(record):
    target = Path(record) / "fitted" / "preprocessors"
    plan = json.loads((target / "plan.json").read_text())
    fitted = {}
    for item in plan["fields"]:
        for name in item["chain"]:
            if name in fitted:
                continue
            with (target / f"{name}.pkl").open("rb") as stream:
                fitted[name] = pickle.load(stream)
    fields = [Field(item["name"], list(item["chain"]), bool(item["target"]), list(item["columns"]),
                    list(item.get("extras") or []))
              for item in plan["fields"]]
    return Prep(fields, fitted, dict(plan["sets"]), dict(plan["dtypes"]), list(plan["drop"]),
                list(plan.get("spectators") or []))


def pil_image(value):
    from PIL import Image

    if not isinstance(value, Image.Image):
        raise TypeError(f"expected an image, got {type(value).__name__}")
    return value


class ImageTensor(Preprocessor):
    dtype = "float32"
    signed = False

    def apply(self, value):
        if isinstance(value, (tuple, list)):
            return torch.stack([self.apply(item) for item in value])
        array = numpy.asarray(pil_image(value), dtype="float32") / 255.0
        if array.ndim == 2:
            array = array[:, :, None]
        tensor = torch.from_numpy(numpy.ascontiguousarray(array.transpose(2, 0, 1)))
        return tensor * 2.0 - 1.0 if self.signed else tensor
