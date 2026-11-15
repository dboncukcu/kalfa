import math
import numbers
from pathlib import Path

import numpy
import pandas

from ...core.errors import KalfaError
from ..reader import fields_of


PREDICTIONS = "predictions.parquet"
IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
               ".bmp": "image/bmp", ".webp": "image/webp"}
IMAGE_SUFFIXES = tuple(IMAGE_TYPES)


def load_table(directory, name=PREDICTIONS):
    path = Path(directory) / name
    if not path.is_file():
        return None
    try:
        return pandas.read_parquet(path)
    except (OSError, ValueError):
        return None


def filtered(frame, where, name=PREDICTIONS):
    if not where:
        return frame
    try:
        return frame.query(where)
    except Exception as exc:
        raise KalfaError(f"bad where for {name}: {exc}") from exc


def read_predictions(directory, name=PREDICTIONS, where=None):
    frame = load_table(directory, name)
    return None if frame is None else filtered(frame, where, name)


def prediction_tables(directory):
    root = Path(directory)
    if not root.is_dir():
        return []
    return sorted(path.name for path in root.glob("predictions*.parquet") if path.is_file())


def read_fitted_fields(directory):
    return fields_of(directory)


def image_column(frame):
    if "path" not in [str(column) for column in frame.columns]:
        return None
    values = [value for value in frame["path"].tolist() if not missing(value)]
    if values and all(isinstance(value, str) and value.lower().endswith(IMAGE_SUFFIXES) for value in values):
        return "path"
    return None


def missing(value):
    if value is None:
        return True
    try:
        return bool(pandas.isna(value))
    except (TypeError, ValueError):
        return False


def label_text(value):
    if isinstance(value, numpy.generic):
        value = value.item()
    if missing(value):
        return None
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        number = float(value)
        if math.isfinite(number) and number.is_integer():
            return str(int(number))
        return format(number, ".12g")
    return str(value)


def labels(classes):
    if not isinstance(classes, (list, tuple)) or not classes:
        return None
    return [label_text(label) or str(label) for label in classes]


def label_key(text):
    try:
        number = float(text)
    except (TypeError, ValueError):
        return 1, 0.0, str(text)
    return (0, number, "") if math.isfinite(number) else (1, 0.0, str(text))


def ordered_labels(values):
    return sorted({value for value in values if value is not None}, key=label_key)


def cached_text(known, value):
    try:
        return known[(type(value), value)]
    except KeyError:
        text = known[(type(value), value)] = label_text(value)
        return text
    except TypeError:
        return label_text(value)


def texts(series):
    known = {}
    return numpy.asarray([cached_text(known, value) for value in series.tolist()], dtype=object)


def present(values):
    return numpy.fromiter((value is not None for value in values), dtype=bool, count=len(values))


def numeric(series):
    return pandas.api.types.is_numeric_dtype(series) and not pandas.api.types.is_bool_dtype(series)


def numbers_of(series):
    return pandas.to_numeric(series, errors="coerce").to_numpy(dtype=numpy.float64)


def finite(values):
    array = numpy.asarray(values, dtype=numpy.float64).reshape(-1)
    return array[numpy.isfinite(array)]


def plain(values):
    return [None if isinstance(value, float) and not math.isfinite(value) else value for value in values.tolist()]


def plain_value(value):
    if isinstance(value, numpy.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return None if missing(value) else value


def rows_in(frame):
    if "row" in frame.columns:
        return frame["row"].to_numpy()
    return frame.index.to_numpy()
