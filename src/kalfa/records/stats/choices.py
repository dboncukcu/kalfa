import math
import numbers

import numpy

from ...core.errors import KalfaError
from ..reader import read_plan
from .classes import (
    COMPLEMENT,
    NEGATED,
    best_threshold,
    binary_rows,
    binary_stats,
    in_unit,
    multiclass_rows,
    multiclass_stats,
    score_values,
)
from .measures import color_column, embedding_stats, regression_stats, values_stats
from .pairs import declared_targets, label_columns, score_labels, task_pairs
from .tables import (
    PREDICTIONS,
    filtered,
    finite,
    image_column,
    label_text,
    load_table,
    numbers_of,
    numeric,
    plain_value,
    present,
    read_fitted_fields,
    rows_in,
    texts,
)


KINDS = {"binary": "classification", "multiclass": "classification", "regression": "regression",
         "embedding": "embedding", "values": "values"}
GALLERY = 24
CHOICES = ("task", "score", "positive", "threshold", "bins")
HISTOGRAMS = {"binary": ("score_hist",), "multiclass": (), "regression": ("residual_hist", "truth_hist"),
              "embedding": ("norm_hist",), "values": ("hist",)}
MAX_BINS = 1000


def entry(frame, position, truth=None, pred=None, score=None, error=None):
    return {"index": int(frame.index[position]), "row": plain_value(rows_in(frame)[position]),
            "truth": None if truth is None else plain_value(truth[position]),
            "pred": None if pred is None else plain_value(pred[position]),
            "score": None if score is None else plain_value(score[position]),
            "error": None if error is None else plain_value(error[position])}


def class_gallery(frame, truth, guess, keep, score=None):
    wrong = keep & (truth != guess).astype(bool)
    picked = numpy.r_[numpy.flatnonzero(wrong), numpy.flatnonzero(keep & ~wrong)][:GALLERY]
    return [entry(frame, position, truth, guess, score) for position in picked]


def class_scores(frame, columns, guess, picked):
    found = numpy.full(len(guess), numpy.nan)
    for position in picked:
        column = columns.get(guess[position])
        if column in frame.columns:
            found[position] = numbers_of(frame[column].iloc[[position]])[0]
    return found


def gallery_of(frame, pair, task, choice):
    pred, target = pair["pred"], pair["target"]
    if task == "binary":
        truth, guess, values, keep = binary_rows(frame, pred, target, choice["labels"], choice["score"],
                                                 choice["threshold"])
        return class_gallery(frame, truth, guess, keep, values)
    if task == "multiclass":
        truth, guess, keep = multiclass_rows(frame, pred, target)
        wrong = keep & (truth != guess).astype(bool)
        picked = numpy.r_[numpy.flatnonzero(wrong), numpy.flatnonzero(keep & ~wrong)][:GALLERY]
        columns = label_columns(pair["scores"], pair["output"], pair["classes"])
        return class_gallery(frame, truth, guess, keep, class_scores(frame, columns, guess, picked))
    if task == "regression":
        guess, truth = numbers_of(frame[pred]), numbers_of(frame[target])
        error = guess - truth
        keep = numpy.flatnonzero(numpy.isfinite(error))
        picked = keep[numpy.argsort(-numpy.abs(error[keep]), kind="stable")][:GALLERY]
        return [entry(frame, position, truth, guess, None, error) for position in picked]
    color = choice.get("color")
    truth = None if color is None else texts(frame[color])
    guess = numbers_of(frame[pred]) if task == "values" else None
    return [entry(frame, position, truth, guess) for position in range(min(len(frame), GALLERY))]


def number_value(value, label):
    try:
        found = float(value)
    except (TypeError, ValueError):
        raise KalfaError(f"{label} must be a number, not {value!r}") from None
    if not math.isfinite(found):
        raise KalfaError(f"{label} must be a finite number, not {value!r}")
    return found


def bins_count(value, label):
    if isinstance(value, bool) or not isinstance(value, (numbers.Integral, str)):
        raise KalfaError(f"{label} must be a whole number, not {value!r}")
    try:
        found = int(value)
    except ValueError:
        raise KalfaError(f"{label} must be a whole number, not {value!r}") from None
    if found < 1:
        raise KalfaError(f"{label} must be at least 1, not {value!r}")
    return min(found, MAX_BINS)


def checked_bins(pred, wanted):
    if wanted in (None, ""):
        return {}
    if not isinstance(wanted, dict):
        raise KalfaError(f"the bins of {pred} must map histogram names to counts, like {{residual_hist: 60}}")
    return {str(name): bins_count(count, f"the bins of {name} for {pred}") for name, count in wanted.items()}


def checked_overrides(overrides, pairs):
    if overrides in (None, {}):
        return {}
    if not isinstance(overrides, dict):
        raise KalfaError("overrides must map prediction columns to choices")
    known = [pair["pred"] for pair in pairs]
    found = {}
    for pred, choice in overrides.items():
        if pred not in known:
            raise KalfaError(f"no prediction pair {pred}; the table has {', '.join(known) or 'none'}")
        if not isinstance(choice, dict):
            raise KalfaError(f"the choice for {pred} must be a mapping of task, score, positive, threshold and bins")
        unknown = sorted(set(choice) - set(CHOICES))
        if unknown:
            raise KalfaError(f"{pred} takes task, score, positive, threshold and bins, not {', '.join(unknown)}")
        found[pred] = {key: value for key, value in choice.items() if key != "bins" and value not in (None, "")}
        bins = checked_bins(pred, choice.get("bins"))
        if bins:
            found[pred]["bins"] = bins
    return found


def chosen_task(pair, choice):
    task = choice.get("task") or pair["auto"]
    if task not in pair["tasks"]:
        raise KalfaError(f"{pair['pred']} cannot be read as {task}; it can be {', '.join(pair['tasks'])}")
    return task


def histogram_bins(pred, task, wanted, bins):
    names = HISTOGRAMS[task]
    unknown = sorted(set(wanted) - set(names))
    if unknown:
        kept = ", ".join(names) or "none"
        raise KalfaError(f"{pred} read as {task} has no histogram {', '.join(unknown)}; its histograms are {kept}")
    return {name: wanted.get(name, bins) for name in names}


def score_candidates(frame, pair, positive, labels):
    owners = score_labels(pair["scores"], pair["output"], labels)
    found = [column for column, label in owners.items() if label == positive]
    found += [f"{COMPLEMENT}{column}" if label in labels else column
              for column, label in owners.items() if label != positive]
    grows = list(pair["classes"])[1]
    for column in (pair["raw"], pair["pred"]):
        if column and column not in found and numeric(frame[column]):
            values = numbers_of(frame[column])
            if len(numpy.unique(finite(values))) > 2:
                found.append(column if positive == grows else turned(column, values))
    return found


def turned(column, values):
    return f"{COMPLEMENT}{column}" if in_unit(values) else f"{NEGATED}{column}"


def binary_choice(frame, pair, choice):
    labels = list(pair["classes"])
    positive = labels[1]
    if choice.get("positive") is not None:
        wanted = label_text(choice["positive"])
        if wanted not in labels:
            raise KalfaError(f"positive must be one of {', '.join(labels)} for {pair['pred']}, not {wanted!r}")
        positive = wanted
    labels = [label for label in labels if label != positive] + [positive]
    candidates = score_candidates(frame, pair, positive, labels)
    score = candidates[0] if candidates else None
    if choice.get("score") is not None:
        if choice["score"] not in candidates:
            shown = ", ".join(candidates) or "none"
            raise KalfaError(f"score must be one of {shown} for {pair['pred']}, not {choice['score']!r}")
        score = choice["score"]
    threshold, rule, unit = None, None, True
    if score is not None:
        values = score_values(frame, score)
        unit = in_unit(values)
        if choice.get("threshold") is not None:
            threshold, rule = number_value(choice["threshold"], "threshold"), "given"
        elif unit:
            threshold, rule = 0.5, "0.5"
        else:
            truth = texts(frame[pair["target"]])
            keep = present(truth) & numpy.isfinite(values)
            threshold, rule = best_threshold((truth == positive)[keep].astype(bool), values[keep]), "best_f1"
    return {"labels": labels, "positive": positive, "score": score, "threshold": threshold, "rule": rule,
            "unit": unit, "candidates": candidates}


def headline(task, found):
    if task == "regression":
        return ("r2", found["r2"]) if found.get("r2") is not None else ("rmse", found.get("rmse"))
    if task == "binary":
        return ("auc", found["roc"]["auc"]) if found.get("roc") else ("f1", found.get("f1"))
    if task == "multiclass":
        return "accuracy", found.get("accuracy")
    if task == "embedding":
        explained = [value for value in (found.get("pca") or {}).get("explained") or [] if value is not None]
        return "explained", float(sum(explained)) if explained else None
    return "count", found.get("count")


def overview_of(pairs):
    tasks = {}
    rows = []
    for pair in pairs:
        tasks[pair["task"]] = tasks.get(pair["task"], 0) + 1
        key, value = headline(pair["task"], pair["stats"])
        rows.append({"output": pair["output"], "pred": pair["pred"], "target": pair["target"], "task": pair["task"],
                     "key": key, "value": value, "count": pair["stats"].get("count")})
    return {"pairs": len(pairs), "tasks": tasks, "rows": rows}


def scored(full, frame, base, choice, images, context, bins, sample, worst):
    task = chosen_task(base, choice)
    counts = histogram_bins(base["pred"], task, choice.get("bins") or {}, bins)
    pair = {"output": base["output"], "pred": base["pred"], "columns": list(base["columns"]),
            "target": base["target"], "kind": KINDS[task], "task": task, "auto": base["auto"],
            "tasks": list(base["tasks"]), "raw": base["raw"], "scores": list(base["scores"]), "score": None,
            "positive": None, "images": images}
    details = {}
    if task == "binary":
        details = binary_choice(full, base, choice)
        pair["score"], pair["positive"] = details["score"], details["positive"]
        stats = binary_stats(frame, base["pred"], base["target"], details["labels"], details["score"],
                             details["threshold"], details["rule"], details["unit"], counts["score_hist"],
                             details["candidates"])
    elif task == "multiclass":
        stats = multiclass_stats(frame, base["pred"], base["target"], base["classes"], base["scores"])
    elif task == "regression":
        stats = regression_stats(frame, base["pred"], base["target"], bins=counts["truth_hist"], sample=sample,
                                 worst=worst, residual_bins=counts["residual_hist"])
    elif task == "embedding":
        details = {"color": color_column(full, context["targets"], images)}
        stats = embedding_stats(frame, base["columns"], details["color"], bins=counts["norm_hist"], sample=sample)
    else:
        stats = values_stats(frame, base["pred"], bins=counts["hist"])
    if images:
        stats["gallery"] = gallery_of(frame, base, task, details)
    pair["stats"] = stats
    return pair


def prediction_stats(directory, plan=None, name=PREDICTIONS, where=None, bins=40, sample=2000, worst=20,
                     overrides=None):
    full = load_table(directory, name)
    if full is None:
        return None
    full = full.reset_index(drop=True)
    frame = filtered(full, where, name)
    if plan is None:
        plan = read_plan(directory)
    fitted = read_fitted_fields(directory)
    bases = task_pairs(full, plan, fitted)
    choices = checked_overrides(overrides, bases)
    images = image_column(full)
    context = {"targets": declared_targets(plan, fitted, full)}
    pairs = [scored(full, frame, base, choices.get(base["pred"]) or {}, images, context, bins, sample, worst)
             for base in bases]
    return {"rows": int(len(frame)), "pairs": pairs, "overview": overview_of(pairs)}
