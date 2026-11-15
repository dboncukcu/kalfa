from pathlib import Path

import pandas

from kalfa.core import aliases
from kalfa.core.errors import BuildError
from kalfa.std.data import splits
from kalfa.std.data.columns import target_of


SET_NAMES = ("train", "valid", "test")
FILTER_KEYS = ("all",) + SET_NAMES
BATCH_DEFAULTS = {"size": 128, "eval_size": None, "workers": 0, "drop_last": False, "shuffle": True}
STREAM_SPLIT = "a stream needs a sequential or given split; stream cuts the file in order"


def batch_spec(batch):
    if isinstance(batch, bool):
        raise BuildError(f"batch is an integer size or a mapping of {list(BATCH_DEFAULTS)}, got {batch!r}")
    if isinstance(batch, int):
        spec = {**BATCH_DEFAULTS, "size": batch}
    elif isinstance(batch, dict):
        unknown = sorted(set(batch) - set(BATCH_DEFAULTS))
        if unknown:
            raise BuildError(f"batch: unknown keys {unknown}; the keys are {list(BATCH_DEFAULTS)}")
        spec = {**BATCH_DEFAULTS, **batch}
    else:
        raise BuildError(f"batch is an integer size or a mapping of {list(BATCH_DEFAULTS)}, got "
                         f"{type(batch).__name__}")
    for key in ("size", "eval_size"):
        value = spec[key]
        if key == "eval_size" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise BuildError(f"batch.{key} must be a positive integer, got {value!r}")
    spec["workers"] = int(spec["workers"] or 0)
    spec["drop_last"] = bool(spec["drop_last"])
    spec["shuffle"] = bool(spec["shuffle"])
    return spec


def batch_of(batch, size=BATCH_DEFAULTS["size"]):
    if isinstance(batch, dict) and "size" not in batch:
        batch = {**batch, "size": size}
    return batch_spec(batch)


def filters_of(spec):
    if spec is None:
        return None, {}
    if isinstance(spec, str):
        return spec, {}
    if isinstance(spec, dict):
        unknown = sorted(set(spec) - set(FILTER_KEYS))
        if unknown:
            raise BuildError(f"filter: unknown keys {unknown}; the keys are {list(FILTER_KEYS)}")
        wrong = [key for key, text in spec.items() if not isinstance(text, str)]
        if wrong:
            raise BuildError(f"filter: {wrong} must be pandas query strings")
        return spec.get("all"), {name: text for name, text in spec.items() if name != "all"}
    raise BuildError("filter is a pandas query string or a mapping of all, train, valid and test to query strings")


def query(frame, text, where):
    try:
        return frame.query(text)
    except Exception as error:
        raise BuildError(f"{where}: the query {text!r} failed: {error}") from error


def split_of(spec):
    if spec is None:
        raise BuildError("data.split is required: a list of ratios such as [0.7, 0.15, 0.15] or a call such as "
                         "{use: kfold, folds: 5, fold: 0}")
    if isinstance(spec, (list, tuple)):
        return splits.RandomSplit(spec)
    if isinstance(spec, str):
        return target_of(spec, "split")()
    if isinstance(spec, dict):
        if "use" not in spec:
            raise BuildError(f"split: a mapping needs use, got the keys {sorted(spec)}")
        params = {key: value for key, value in spec.items() if key != "use"}
        return target_of(spec["use"], "split")(**params)
    if hasattr(spec, "split"):
        return spec
    raise BuildError(f"split is a list of ratios, a call or an object with split(frame), got {type(spec).__name__}")


def split_check(spec):
    if isinstance(spec, (list, tuple)):
        splits.RandomSplit(spec)


def split_target(spec):
    if isinstance(spec, (list, tuple)):
        return splits.RandomSplit
    if isinstance(spec, str):
        return target_of(spec, "split")
    if isinstance(spec, dict):
        return target_of(spec["use"], "split") if "use" in spec else None
    if hasattr(spec, "split"):
        return type(spec)
    return getattr(spec, "target", None)


def stream_split_check(stream, split):
    if not stream or split is None:
        return
    try:
        target = split_target(split)
    except BuildError:
        return
    if isinstance(target, type) and not issubclass(target, (splits.SequentialSplit, splits.GivenSplit)):
        raise BuildError(STREAM_SPLIT)


def stream_check(value):
    if value is None or value is False:
        return value
    from kalfa.std.data.stream import options_of

    return options_of(value)


def locate(file, directories=()):
    path = Path(str(file)).expanduser()
    if path.exists():
        return path
    searched = list(dict.fromkeys(str(item) for item in [*(directories or ()), *reversed(aliases.SEARCH_DIRS)]))
    if not path.is_absolute():
        for directory in searched:
            candidate = Path(directory) / path
            if candidate.exists():
                return candidate
    looked = "" if path.is_absolute() or not searched else f"; also looked under {[str(item) for item in searched]}"
    raise BuildError(f"data file {str(file)!r} not found{looked}")


def read_table(path):
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        return pandas.read_parquet(path)
    if suffix == ".csv":
        return pandas.read_csv(path)
    if suffix == ".tsv":
        return pandas.read_csv(path, sep="\t")
    raise BuildError(f"{path}: the table format {suffix!r} is unknown; use .parquet or .csv")
