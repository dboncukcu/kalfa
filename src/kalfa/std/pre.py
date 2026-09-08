"""Preprocessing: the field plan, fitting on the train set, applying per set, and the std preprocessors."""

import copy
import fnmatch
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy

from ..registration import lego
from ..kinds import SETS
from .samples import is_samples
from .stream import is_stream

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
                values = _run_chain(item, chunk[item.name].to_numpy(), self.prep.fitted, {}, self.sets, self.set_name,
                                    fit=False)
                if len(values):
                    values, _ = _cast(values, item.name)
                if values.ndim == 1:
                    columns[item.columns[0]] = values
                else:
                    for position, column in enumerate(item.columns):
                        columns[column] = values[:, position]
            yield numpy.asarray(chunk.index), pandas.DataFrame(columns, index=chunk.index)


def _columns_of_source(df):
    if is_samples(df):
        return list(df.fields)
    return list(df.columns)


def _fit_stream(items, df, templates, sets, fitted, dtypes):
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
        values = _run_chain(item, values, fitted, templates, sets, "train", fit=False)
        if len(values):
            values, kind = _cast(values, item.name)
        else:
            kind = torch_dtype(values.dtype) or "float32"
        item.columns = _columns_of(item, values, fitted)
        for column in item.columns:
            dtypes[column] = kind


def _resolve(df, fields, drop):
    columns = [name for name in _columns_of_source(df) if name not in (drop or [])]
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


def _fit_chains(items, values_of, templates, sets, fitted, set_name="train"):
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


def _run_chain(item, values, fitted, templates, sets, set_name, fit):
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


def _columns_of(item, values, fitted):
    if values.ndim == 1:
        return [item.name]
    for pre in reversed(item.chain):
        obj = fitted_object(fitted, pre, item.name)
        if obj is not None and hasattr(obj, "columns"):
            return list(obj.columns(item.name))
    return [f"{item.name}_{position}" for position in range(values.shape[1])]


def _cast(values, name):
    kind = torch_dtype(values.dtype)
    if kind is None:
        raise TypeError(f"field {name!r} has dtype {values.dtype} after its chain; add a preprocessor that turns it "
                        f"into a number (cast, one_hot, label_encoder, a tokenizer)")
    return values.astype(kind), kind


def _values(df, name):
    return df[name].to_numpy()


def sets_of(keys):
    """The sets each preprocessor applies to, from the definition keys; a missing entry means every set."""
    found = {}
    for name, entry in (keys or {}).items():
        if isinstance(entry, dict) and entry.get("sets") is not None:
            found[name] = list(entry["sets"])
    return found


@lego("/lego/kalfa/fit", returns="prep", state=True, bus=["record"],
            description="Resolve the field globs and fit every preprocessor chain on the train set; keys carry the "
                        "sets a preprocessor is limited to")
def fit(df, fields, preprocessors, drop, keys=None, record=None):
    sets = sets_of(keys)
    templates = dict(preprocessors or {})
    unknown = sorted({pre for spec in (fields or {}).values() for pre in ((spec or {}).get("preprocessors") or [])
                      if pre not in templates})
    if unknown:
        raise ValueError(f"fields name preprocessors {unknown} that data.preprocessors does not define")
    items = _resolve(df, fields or {}, drop)
    sets = {name: list(allowed) for name, allowed in (sets or {}).items()}
    fitted = {}
    dtypes = {}
    if is_stream(df):
        _fit_stream(items, df, templates, sets, fitted, dtypes)
        prep = Prep(items, fitted, sets, dtypes, list(drop or []))
        if record is not None:
            write_prep(prep, record)
        return prep
    if is_samples(df):
        for item in items:
            kind = df.dtypes.get(item.name)
            for pre in item.chain:
                obj = copy.deepcopy(templates[pre])
                if hasattr(obj, "fit"):
                    try:
                        values = df.column(item.name)
                    except ValueError:
                        raise ValueError(f"preprocessor {pre!r} fits on a column, but field {item.name!r} of the "
                                         f"Dataset source cannot be read as one") from None
                    obj.fit(values)
                fitted.setdefault(pre, {})[item.name] = obj
                kind = getattr(obj, "dtype", kind)
            item.columns = [item.name]
            resolved = torch_dtype(kind)
            if resolved is None:
                raise TypeError(f"field {item.name!r} has dtype {kind} after its chain; add a preprocessor that turns "
                                f"it into a tensor (to_tensor)")
            dtypes[item.name] = resolved
        prep = Prep(items, fitted, sets, dtypes, list(drop or []))
        if record is not None:
            write_prep(prep, record)
        return prep
    final = _fit_chains(items, lambda item: _values(df, item.name), templates, sets, fitted)
    for item in items:
        values = final[item.name]
        if len(values):
            values, kind = _cast(values, item.name)
        else:
            kind = torch_dtype(values.dtype) or "float32"
        item.columns = _columns_of(item, values, fitted)
        for column in item.columns:
            dtypes[column] = kind
    prep = Prep(items, fitted, sets, dtypes, list(drop or []))
    if record is not None:
        write_prep(prep, record)
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


@lego("/lego/kalfa/apply",
            description="Apply the fitted chains to one set and type its columns; keys carry the sets a "
                        "preprocessor is limited to")
def apply(df, prep, set, keys=None):
    import pandas

    sets = sets_of(keys) if keys is not None else prep.sets
    if is_stream(df):
        missing = [item.name for item in prep.fields if item.name not in df.columns]
        if missing:
            raise ValueError(f"the {set} data lacks columns {missing}")
        return Frame(None, prep.features, prep.targets, set, stream=StreamView(df, prep, set, sets))
    if is_samples(df):
        chains = {}
        for item in prep.fields:
            if item.name not in df.fields:
                raise ValueError(f"the {set} data lacks field {item.name!r}")
            chains[item.name] = [prep.object_of(pre, item.name) for pre in item.chain
                                 if sets.get(pre) is None or set in sets[pre]]
        return Frame(None, [], prep.targets, set, None, df, [item.name for item in prep.fields], chains)
    columns = {}
    for item in prep.fields:
        if item.name not in df.columns:
            raise ValueError(f"the {set} data lacks column {item.name!r}")
        values = _values(df, item.name)
        values = _run_chain(item, values, prep.fitted, {}, sets, set, fit=False)
        if len(values):
            values, _ = _cast(values, item.name)
        if values.ndim == 1:
            columns[item.columns[0]] = values
        else:
            for position, column in enumerate(item.columns):
                columns[column] = values[:, position]
    data = pandas.DataFrame(columns, index=df.index)
    if len(data) == 0:
        data = pandas.DataFrame({column: numpy.zeros(0, dtype=prep.dtypes[column]) for column in prep.dtypes},
                                index=df.index)
    named = {item.name for item in prep.fields}
    extra = df[[column for column in df.columns if column not in named and column not in prep.drop]]
    return Frame(data, prep.features, prep.targets, set, extra)


class _Scaler:
    """A sklearn scaler over every column that names it: one object, per column statistics (``grouped``).

    ``fit`` takes the matrix of the columns in plan order; ``apply`` and ``inverse`` take the whole matrix, or a
    slice of it with ``columns`` naming the positions the slice holds.
    """

    rescales = True
    grouped = True

    def build(self):
        raise NotImplementedError

    def fit(self, values):
        self.scaler = self.build().fit(_block(values))

    def partial_fit(self, values):
        if getattr(self, "scaler", None) is None:
            self.scaler = self.build()
        self.scaler.partial_fit(_block(values))

    def apply(self, values, columns=None):
        return self._run(values, columns, forward=True)

    def inverse(self, values, columns=None):
        return self._run(values, columns, forward=False)

    def _run(self, values, columns, forward):
        matrix = _block(values)
        width = int(getattr(self.scaler, "n_features_in_", matrix.shape[1]))
        positions = numpy.arange(width) if columns is None else numpy.asarray(columns, dtype=int)
        if len(positions) != matrix.shape[1]:
            raise ValueError(f"the scaler was fitted on {width} columns and got {matrix.shape[1]}")
        shift, scale = self.terms()
        out = (matrix - shift[positions]) / scale[positions] if forward else matrix * scale[positions] + shift[positions]
        return numpy.asarray(out).reshape(numpy.asarray(values).shape)

    def terms(self):
        """The per column (shift, scale) of the fitted scaler: apply is (value - shift) / scale."""
        raise NotImplementedError


def _block(values):
    matrix = numpy.asarray(values, dtype="float64")
    return matrix.reshape(-1, 1) if matrix.ndim == 1 else matrix


class StandardScaler(_Scaler):
    def build(self):
        from sklearn.preprocessing import StandardScaler as Scaler

        return Scaler()

    def terms(self):
        return numpy.asarray(self.scaler.mean_, dtype="float64"), numpy.asarray(self.scaler.scale_, dtype="float64")


@lego("/pre/sklearn/standard_scaler", state=True, alias="standard_scaler", grouped=True,
            description="Standardize a column to zero mean and unit variance (sklearn StandardScaler); one object "
                        "over every column that names it, its statistics per column")
def standard_scaler():
    return StandardScaler()


class MinMaxScaler(_Scaler):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def build(self):
        from sklearn.preprocessing import MinMaxScaler as Scaler

        return Scaler(feature_range=(self.low, self.high))

    def terms(self):
        scale = numpy.asarray(self.scaler.scale_, dtype="float64")
        return -numpy.asarray(self.scaler.min_, dtype="float64") / scale, 1.0 / scale


@lego("/pre/sklearn/minmax_scaler", state=True, alias="minmax_scaler", grouped=True,
            description="Scale a column into [low, high] (sklearn MinMaxScaler); one object over every column that "
                        "names it, its statistics per column")
def minmax_scaler(low=0.0, high=1.0):
    return MinMaxScaler(low, high)


class Cast:
    def __init__(self, dtype):
        self.dtype = dtype

    def apply(self, values):
        return numpy.asarray(values).astype(self.dtype)


@lego("/pre/kalfa/cast", alias="cast", description="Cast a column to a numpy dtype")
def cast(dtype):
    return Cast(dtype)


class Absolute:
    def apply(self, values):
        return numpy.abs(numpy.asarray(values))


@lego("/pre/kalfa/abs", alias="abs", description="Absolute value of a column")
def absolute():
    return Absolute()


class Log:
    rescales = True

    def __init__(self, base, norm):
        self.base = base
        self.norm = norm

    def apply(self, values):
        scaled = numpy.asarray(values, dtype="float64") / self.norm
        return numpy.log1p(scaled) / numpy.log(self.base)

    def inverse(self, values):
        return numpy.expm1(numpy.asarray(values, dtype="float64") * numpy.log(self.base)) * self.norm


@lego("/pre/kalfa/log", alias="log",
            description="log1p of a column divided by norm, in the given base")
def log(base=10.0, norm=1.0):
    return Log(base, norm)


def all_sets():
    return list(SETS)


class OneHot:
    def fit(self, values):
        from sklearn.preprocessing import OneHotEncoder

        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(
            numpy.asarray(values, dtype=object).reshape(-1, 1))
        self.categories = [str(category) for category in self.encoder.categories_[0]]

    def apply(self, values):
        return self.encoder.transform(numpy.asarray(values, dtype=object).reshape(-1, 1)).astype("float32")

    def columns(self, name):
        return [f"{name}_{category}" for category in self.categories]


@lego("/pre/kalfa/one_hot", state=True, alias="one_hot",
            description="One hot columns <field>_<category> of a categorical column; unknown categories give zeros")
def one_hot():
    return OneHot()


class LabelEncoder:
    def fit(self, values):
        self.classes = numpy.array(sorted(set(numpy.asarray(values).tolist())), dtype=object)
        self.lookup = {value: position for position, value in enumerate(self.classes.tolist())}

    def apply(self, values):
        try:
            return numpy.array([self.lookup[value] for value in numpy.asarray(values).tolist()], dtype="int64")
        except KeyError as exc:
            raise ValueError(f"label {exc.args[0]!r} was not seen when the encoder was fitted") from None

    def inverse(self, values):
        codes = numpy.asarray(values).astype("int64").reshape(-1)
        return self.classes[codes]

    def decode(self, scores):
        matrix = numpy.asarray(scores)
        if matrix.ndim == 1 or matrix.shape[1] == 1:
            codes = (matrix.reshape(-1) > 0).astype("int64")
        else:
            codes = matrix.argmax(axis=1)
        return self.classes[codes]


@lego("/pre/kalfa/label_encoder", state=True, alias="label_encoder",
            description="Integer codes of a label column, sorted by label; inverted in reports and predictions, "
                        "class scores decode to labels")
def label_encoder():
    return LabelEncoder()


IMAGE_STATS = {"imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
               "cifar10": ([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])}


def _pil(value):
    from PIL import Image

    if not isinstance(value, Image.Image):
        raise TypeError(f"expected an image, got {type(value).__name__}")
    return value


class ToTensor:
    """PIL image to a float tensor in [0, 1], channels first (grayscale gets one channel)."""

    dtype = "float32"

    def __init__(self, signed=False):
        self.signed = signed

    def apply(self, value):
        import torch

        if isinstance(value, (tuple, list)):
            return torch.stack([self.apply(item) for item in value])
        array = numpy.asarray(_pil(value), dtype="float32") / 255.0
        if array.ndim == 2:
            array = array[:, :, None]
        tensor = torch.from_numpy(numpy.ascontiguousarray(array.transpose(2, 0, 1)))
        return tensor * 2.0 - 1.0 if self.signed else tensor


@lego("/pre/kalfa/to_tensor", alias="to_tensor",
            description="Image to a float tensor in [0, 1], channels first")
def to_tensor():
    return ToTensor()


@lego("/pre/kalfa/to_tensor_signed", alias="to_tensor_signed",
            description="Image to a float tensor in [-1, 1], channels first")
def to_tensor_signed():
    return ToTensor(signed=True)


class Resize:
    def __init__(self, size):
        self.size = (int(size), int(size)) if isinstance(size, (int, float)) else (int(size[1]), int(size[0]))

    def apply(self, value):
        return _pil(value).resize(self.size)


@lego("/pre/kalfa/resize", alias="resize", description="Resize an image to size (int or [h, w])")
def resize(size):
    return Resize(size)


class RandomCropFlip:
    """Pad by four pixels, crop a random size by size window, flip horizontally half of the time (train time
    augmentation, limited to sets with the definition's sets key)."""

    def __init__(self, size, padding=4):
        self.size = int(size)
        self.padding = int(padding)

    def apply(self, value):
        from PIL import Image, ImageOps

        image = ImageOps.expand(_pil(value), border=self.padding, fill=0)
        width, height = image.size
        left = int(numpy.random.randint(0, max(width - self.size, 0) + 1))
        top = int(numpy.random.randint(0, max(height - self.size, 0) + 1))
        image = image.crop((left, top, left + self.size, top + self.size))
        if numpy.random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image


@lego("/pre/kalfa/random_crop_flip", alias="random_crop_flip",
            description="Random crop of size after padding and a random horizontal flip")
def random_crop_flip(size):
    return RandomCropFlip(size)


class Normalize:
    dtype = "float32"

    def __init__(self, mean, std):
        self.mean = IMAGE_STATS[mean][0] if isinstance(mean, str) else mean
        self.std = IMAGE_STATS[std][1] if isinstance(std, str) else std

    def apply(self, value):
        import torch

        channels = value.shape[0]
        mean = torch.as_tensor(self.mean if isinstance(self.mean, list) else [self.mean] * channels,
                               dtype=value.dtype).reshape(-1, 1, 1)
        std = torch.as_tensor(self.std if isinstance(self.std, list) else [self.std] * channels,
                              dtype=value.dtype).reshape(-1, 1, 1)
        return (value - mean) / std


@lego("/pre/kalfa/normalize", alias="normalize",
            description="Normalize an image tensor per channel; mean and std are numbers, lists or the presets "
                        "imagenet and cifar10")
def normalize(mean, std):
    return Normalize(mean, std)


class TwoViews:
    def __init__(self, transform):
        self.transform = transform

    def apply(self, value):
        return (self.transform.apply(value), self.transform.apply(value))


@lego("/pre/kalfa/two_views", alias="two_views", refs={"transform": "preprocessor"},
            description="Two independent applications of a transform to one image, as a pair")
def two_views(transform):
    return TwoViews(transform)


class SimclrAug:
    """A random resized crop to size, a random horizontal flip and a brightness jitter."""

    def __init__(self, size, scale=(0.5, 1.0)):
        self.size = int(size)
        self.scale = scale

    def apply(self, value):
        from PIL import Image, ImageEnhance

        image = _pil(value)
        width, height = image.size
        fraction = float(numpy.random.uniform(self.scale[0], self.scale[1]))
        crop_width = max(1, int(round(width * fraction)))
        crop_height = max(1, int(round(height * fraction)))
        left = int(numpy.random.randint(0, width - crop_width + 1))
        top = int(numpy.random.randint(0, height - crop_height + 1))
        image = image.crop((left, top, left + crop_width, top + crop_height)).resize((self.size, self.size))
        if numpy.random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return ImageEnhance.Brightness(image).enhance(float(numpy.random.uniform(0.6, 1.4)))


@lego("/pre/kalfa/simclr_aug", alias="simclr_aug",
            description="SimCLR augmentation: random resized crop to size, horizontal flip, brightness jitter")
def simclr_aug(size, scale=(0.5, 1.0)):
    return SimclrAug(size, tuple(scale))


class CharTokenizer:
    """Characters to ids in sorted order, fitted on the train text; the newline is always in the vocabulary."""

    dtype = "int64"

    def fit(self, values):
        chars = {"\n"}
        for text in values:
            chars.update(str(text))
        self.chars = sorted(chars)
        self.lookup = {char: position for position, char in enumerate(self.chars)}

    def encode(self, text):
        unknown = self.lookup.get(" ", 0)
        return numpy.array([self.lookup.get(char, unknown) for char in str(text)], dtype="int64")

    def apply(self, value):
        if isinstance(value, numpy.ndarray) and value.dtype != object and value.ndim == 1 and value.dtype.kind == "i":
            return value
        return self.encode(value)

    def decode(self, ids):
        return "".join(self.chars[int(position)] for position in numpy.asarray(ids).reshape(-1))

    @property
    def size(self):
        return len(self.chars)


@lego("/pre/kalfa/char_tokenizer", state=True, alias="char_tokenizer",
            description="Character level tokenizer fitted on the train text; the vocabulary goes into the record")
def char_tokenizer():
    return CharTokenizer()
