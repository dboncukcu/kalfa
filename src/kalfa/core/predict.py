import fnmatch

import numpy
import pandas
import torch

from .errors import KalfaError, RunError
from .record import atomic


WIDE = 32


def batch_size(batch):
    if isinstance(batch, dict):
        values = batch.values()
    elif isinstance(batch, (list, tuple)):
        values = batch
    else:
        values = [batch]
    for value in values:
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape):
            return int(shape[0])
    return 1


def enter(ctx, batch):
    ctx.batches += 1
    return batch if ctx.device is None else ctx.device.move(batch)


def empty(loader):
    try:
        return len(loader) == 0
    except TypeError:
        return False


def as_matrix(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = numpy.asarray(value)
    if value.ndim == 0:
        value = value.reshape(1)
    return value.reshape(len(value), -1)


def spread(label, values):
    values = numpy.asarray(values)
    if values.ndim <= 1:
        return {label: values}
    if values.shape[1] == 1:
        return {label: values[:, 0]}
    return {f"{label}_{position}": values[:, position] for position in range(values.shape[1])}


def target_names(data, batch=None):
    names = [str(name) for name in getattr(data, "target_names", None) or ()]
    return names if batch is None else [name for name in names if name in batch]


def is_label(matrix):
    return bool(numpy.issubdtype(matrix.dtype, numpy.integer) or matrix.dtype == bool)


def labelled(data, name, truths):
    truth = truths.get(name)
    if truth is not None:
        return is_label(truth)
    asks = getattr(data, "is_label", None)
    return bool(asks(name)) if callable(asks) else False


def rescales(data, name):
    asks = getattr(data, "rescales", None)
    return bool(asks(name)) if callable(asks) else False


def inverse(data, name, matrix):
    transform = getattr(data, "inverse", None)
    if not callable(transform):
        return matrix
    values = matrix[:, 0] if matrix.ndim == 2 and matrix.shape[1] == 1 else matrix
    return numpy.asarray(transform(name, values))


def decoded(data, name, indexes):
    decode = getattr(data, "decode", None)
    if not callable(decode):
        return None
    return numpy.asarray(decode(name, indexes))


def pairings(ctx, outputs, targets):
    pairs = {}
    for entry in [*ctx.losses.values(), *ctx.metrics.values()]:
        pred, target = getattr(entry, "pred", None), getattr(entry, "target", None)
        if pred in outputs and isinstance(target, str):
            pairs.setdefault(pred, target)
    for name in outputs:
        if name in targets:
            pairs.setdefault(name, name)
    if len(targets) == 1:
        for name in outputs:
            pairs.setdefault(name, targets[0])
    return pairs


def rows_of(data, set_name, count, loader=None, own=True):
    rows = getattr(getattr(loader, "dataset", None), "rows", None)
    if rows is not None and len(rows) == count:
        return numpy.asarray(rows)
    rows = getattr(data, "rows", None)
    if own and callable(rows):
        try:
            found = numpy.asarray(rows(set_name))
        except KalfaError:
            found = None
        if found is not None and len(found) == count:
            return found
    return numpy.arange(count)


def aligned(frame, rows, count):
    if frame is None or len(frame) == count:
        return frame
    try:
        picked = frame.loc[list(rows)]
    except KeyError:
        picked = None
    if picked is None or len(picked) != count:
        raise RunError(f"{count} predicted rows cannot be aligned with the {len(frame)} rows of the frame; the "
                       f"frame needs a unique index")
    return picked


def spectator_columns(data, set_name, count, frame=None):
    if frame is not None:
        patterns = [str(pattern) for pattern in getattr(data, "spectator_patterns", None) or ()]
        return {column: frame[column].to_numpy() for column in frame.columns
                if any(fnmatch.fnmatchcase(str(column), pattern) for pattern in patterns)}
    extra = data.spectators(set_name) if callable(getattr(data, "spectators", None)) else None
    if extra is None:
        return {}
    if isinstance(extra, pandas.DataFrame):
        extra = {column: extra[column].to_numpy() for column in extra.columns}
    return {name: numpy.asarray(values) for name, values in dict(extra).items() if len(values) == count}


def collect_outputs(ctx, loader, entry=None):
    data = ctx.data
    names = None
    guesses, truths = {}, {}
    with torch.no_grad():
        for batch in loader:
            batch = enter(ctx, batch)
            if names is None:
                names = target_names(data, batch)
            outputs = entry.call(batch) if entry is not None else ctx.step.predict(batch, ctx)
            for name, value in outputs.items():
                guesses.setdefault(name, []).append(as_matrix(value))
            for name in names:
                truths.setdefault(name, []).append(as_matrix(batch[name]))
    guesses = {name: numpy.concatenate(pieces) for name, pieces in guesses.items()}
    truths = {name: numpy.concatenate(pieces) for name, pieces in truths.items()}
    return guesses, truths


def spread_truth(columns, data, name, matrix):
    columns.update(spread(name, inverse(data, name, matrix) if rescales(data, name) else matrix))


def truth_columns(columns, data, truths, frame):
    if frame is not None:
        for name in target_names(data):
            matrix = truths.get(name)
            if matrix is not None and matrix.shape[1] > 1 and not is_label(matrix):
                spread_truth(columns, data, name, matrix)
            elif name in frame.columns:
                columns[name] = frame[name].to_numpy()
        return
    for name, matrix in truths.items():
        if is_label(matrix):
            indexes = matrix[:, 0] if matrix.shape[1] == 1 else matrix.argmax(axis=1)
            labels = decoded(data, name, indexes)
            columns[name] = indexes if labels is None else labels
        else:
            spread_truth(columns, data, name, matrix)


def softmax(matrix):
    shifted = matrix.astype(numpy.float64) - matrix.max(axis=1, keepdims=True)
    weights = numpy.exp(shifted)
    return weights / weights.sum(axis=1, keepdims=True)


def sigmoid(values):
    return 0.5 * (1.0 + numpy.tanh(0.5 * values.astype(numpy.float64)))


def class_names(data, target, count):
    indexes = numpy.arange(count)
    try:
        labels = decoded(data, target, indexes)
    except (KalfaError, IndexError, KeyError, ValueError):
        labels = None
    if labels is None or len(labels) != count:
        return [str(index) for index in indexes]
    return [str(label) for label in labels]


def class_scores(data, target, width):
    if width == 1:
        return True
    try:
        labels = decoded(data, target, numpy.arange(width))
    except (KalfaError, IndexError, KeyError, ValueError):
        return False
    return labels is None or len(labels) == width


def probability_columns(columns, data, name, target, matrix):
    if matrix.shape[1] > 1:
        probabilities = softmax(matrix)
        for position, label in enumerate(class_names(data, target, matrix.shape[1])):
            columns[f"prob_{name}_{label}"] = probabilities[:, position]
    else:
        positive = class_names(data, target, 2)[1]
        columns[f"prob_{name}_{positive}"] = sigmoid(matrix[:, 0])


def guess_columns(columns, ctx, guesses, truths):
    data = ctx.data
    targets = target_names(data) or list(truths)
    pairs = pairings(ctx, list(guesses), targets)
    for name, matrix in guesses.items():
        target = pairs.get(name)
        if target is not None and target not in targets:
            target = None
        if target is not None and labelled(data, target, truths) and not class_scores(data, target, matrix.shape[1]):
            target = None
        if target is None:
            if matrix.shape[1] > WIDE:
                continue
            columns.update(spread(f"pred_{name}", matrix))
        elif labelled(data, target, truths):
            indexes = matrix.argmax(axis=1) if matrix.shape[1] > 1 else (matrix[:, 0] > 0).astype(numpy.int64)
            labels = decoded(data, target, indexes)
            columns[f"pred_{name}"] = indexes if labels is None else labels
            if labels is not None:
                columns[f"raw_{name}"] = indexes
            probability_columns(columns, data, name, target, matrix)
        elif rescales(data, target):
            columns.update(spread(f"pred_{name}", inverse(data, target, matrix)))
            columns.update(spread(f"raw_{name}", matrix))
        else:
            columns.update(spread(f"pred_{name}", matrix))


def predictions_table(ctx, set_name="test", loader=None, frame=None, entry=None):
    own = loader is None
    if own:
        loader = ctx.loaders().get(set_name)
    if loader is None or empty(loader):
        return pandas.DataFrame()
    ctx.set_modes(False)
    ctx.set_name = set_name
    data = ctx.data
    guesses, truths = collect_outputs(ctx, loader, entry)
    if not guesses:
        return pandas.DataFrame()
    count = len(next(iter(guesses.values())))
    rows = rows_of(data, set_name, count, loader, own)
    frame = aligned(frame, rows, count)
    columns = {"row": rows}
    truth_columns(columns, data, truths, frame)
    guess_columns(columns, ctx, guesses, truths)
    for name, values in spectator_columns(data, set_name, count, frame).items():
        columns.setdefault(name, values)
    return pandas.DataFrame(columns)


def write_predictions(ctx, set_name):
    table = predictions_table(ctx, set_name)
    if ctx.record is not None and len(table):
        name = "predictions.parquet" if set_name == "test" else f"predictions_{set_name}.parquet"
        with atomic(ctx.record.path(name)) as temporary:
            table.to_parquet(temporary, index=False)
    return table
