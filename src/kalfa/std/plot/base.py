import numpy

from kalfa.std.common import figure
from pathlib import Path
import fnmatch
import pandas


def true_column(pred, targets):
    matches = [target for target in targets if pred == f"pred_{target}" or pred.endswith(f"_{target}")]
    return max(matches, key=len) if matches else None


def panel_title(pred, target, paired):
    if paired.count(target) < 2 or pred == f"pred_{target}" or not pred.endswith(f"_{target}"):
        return target
    return f"{target} ({pred[len('pred_'):-len(target) - 1]})"


def prediction_pairs(predictions):
    if predictions is None or len(predictions) == 0:
        return []
    preds = [column for column in predictions.columns if column.startswith("pred_")]
    targets = [column for column in predictions.columns
               if not column.startswith(("pred_", "raw_")) and column != "row"]
    if not preds or not targets:
        return []
    pairs = [(pred, true_column(pred, targets)) for pred in preds]
    pairs = [(pred, true) for pred, true in pairs if true is not None]
    if not pairs:
        pairs = list(zip(preds, targets * len(preds)))
    return [(pred, true) for pred, true in pairs
            if predictions[pred].dtype.kind in "fiu" and predictions[true].dtype.kind in "fiu"]


def r2_of(true, guess):
    spread = float(numpy.sum((true - numpy.mean(true)) ** 2))
    if spread <= 0:
        return float("nan")
    return 1.0 - float(numpy.sum((guess - true) ** 2)) / spread


def report_loader(loaders, set_name=None):
    names = [set_name] if set_name else ["test", "valid"]
    for name in names:
        loader = (loaders or {}).get(name)
        if loader is not None and loader.dataset.size():
            return name, loader
    return None, None


def scores_and_labels(predictions):
    preds = [column for column in predictions.columns if column.startswith("raw_")]
    targets = [column for column in predictions.columns
               if not column.startswith(("pred_", "raw_")) and column != "row"]
    if not preds or not targets:
        return None, None
    return predictions[preds[0]].to_numpy(dtype="float64"), predictions[targets[0]].to_numpy()


def turn_files(record, suffix):
    return sorted((Path(record) / "samples").glob(f"turn_*{suffix}"))


def first_set(sets, default="train"):
    names = [name for name in (sets or []) if name]
    return names[0] if names else default


def set_frame(loaders, prep, set_name="train"):
    loader = (loaders or {}).get(set_name)
    frame = loader.dataset.frame if loader is not None else None
    data = frame.data if frame is not None else None
    if prep is None or data is None or not len(data):
        return None
    features = [column for column in prep.features]
    if features and all(column in data.columns for column in features):
        table = pandas.DataFrame(prep.rescale_features(data[features].to_numpy(dtype="float64"), set_name),
                                 columns=features, index=data.index)
    else:
        table = pandas.DataFrame(index=data.index)
    for target, columns in (frame.targets or {}).items():
        if len(columns) == 1 and columns[0] in data.columns:
            table[target] = prep.inverse(target, data[columns[0]].to_numpy(dtype="float64"), set_name)
    extra = frame.extra
    for column in (list(extra.columns) if extra is not None else []):
        if column not in table.columns:
            table[column] = extra[column].to_numpy()
    return table if len(table.columns) else None


def columns_of(table, patterns=None, skip=()):
    names = [column for column in table.columns
             if column not in skip and table[column].dtype.kind in "fiub"]
    if not patterns:
        return names
    wanted = [patterns] if isinstance(patterns, str) else list(patterns)
    picked = []
    for pattern in wanted:
        glob = any(character in str(pattern) for character in "*?[")
        for column in names:
            hit = fnmatch.fnmatchcase(column, str(pattern)) if glob else column == pattern
            if hit and column not in picked:
                picked.append(column)
    return picked


def logged(values, column, log):
    if column not in set(log or ()):
        return values, column
    floor = numpy.nanpercentile(numpy.abs(values[values != 0]), 1) if numpy.any(values != 0) else 1e-12
    return numpy.log10(numpy.clip(values, max(float(floor), 1e-300), None)), f"log10({column})"


def pick_pair(predictions, output=None, target=None):
    pairs = prediction_pairs(predictions)
    for pred, field in pairs:
        if output is not None and not (pred == f"pred_{output}" or pred.startswith(f"pred_{output}_")):
            continue
        if target is not None and field != target:
            continue
        return pred, field
    if output is None and target is None and pairs:
        return pairs[0]
    return None, None


def bars(axis, names, values, groups=None, spread=None):
    table = dict(groups or {})
    labels = [table.get(column, "") for column in names]
    ordered = [label for label in dict.fromkeys(labels) if label]
    colors = {label: figure.CATEGORICAL[position % len(figure.CATEGORICAL)]
              for position, label in enumerate(ordered)}
    painted = [colors.get(label, figure.CATEGORICAL[0]) for label in labels]
    axis.barh(names, values, xerr=spread, height=0.66, color=painted,
              error_kw={"ecolor": figure.INK_MUTED, "elinewidth": 1} if spread is not None else None)
    axis.axvline(0.0, color=figure.INK_MUTED, linewidth=1)
    axis.grid(axis="y", visible=False)
    if len(ordered) > 1:
        handles = [figure.pyplot().Line2D([], [], marker="s", linestyle="", markersize=8, color=colors[label],
                                          label=label) for label in ordered]
        axis.legend(handles=handles, loc="lower right")
    return ordered
