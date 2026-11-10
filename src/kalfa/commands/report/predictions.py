from pathlib import Path

from kalfa.commands.report.page import color_of, embedded, escape, finite, new_figure, paragraph, plural, shown, table
from kalfa.core.errors import KalfaError
from kalfa.records import reader
from kalfa.records.stats import PREDICTIONS, prediction_stats
from kalfa.std.plots import INK, LEGEND_LIMIT, MAX_CLASSES, MUTED, dress


ANNOTATED = 16
THRESHOLD_RULES = {"0.5": "the threshold 0.5 of a probability", "given": "a threshold given by hand",
                   "best_f1": "the threshold with the best f1 on this table"}


def regression_part(pair, found):
    keys = ("count", "rmse", "mae", "r2", "bias")
    parts = [table(keys, [[found.get(key) for key in keys]])]
    scatter = found.get("scatter") or {}
    points = [(truth, guess) for truth, guess in zip(scatter.get("truth") or [], scatter.get("pred") or [])
              if finite(truth) and finite(guess)]
    if not points:
        parts.append(paragraph("no finite rows to draw", "muted"))
        return "\n".join(parts)
    figure, (left, right) = new_figure(9.6, 3.8, columns=2)
    truths, guesses = zip(*points)
    left.scatter(truths, guesses, s=10, alpha=0.6, color=color_of(0), linewidths=0)
    low, high = min(min(truths), min(guesses)), max(max(truths), max(guesses))
    left.plot([low, high], [low, high], color=MUTED, linestyle="--", linewidth=1)
    left.set_title("pred against truth", loc="left")
    left.set_xlabel(pair["target"])
    left.set_ylabel(pair["pred"])
    dress(left)
    for position, (key, label) in enumerate((("truth_hist", pair["target"]), ("pred_hist", pair["pred"]))):
        counted = found.get(key) or {}
        if counted.get("counts") and len(counted.get("edges") or []) == len(counted["counts"]) + 1:
            right.stairs(counted["counts"], counted["edges"], label=label, color=color_of(position), linewidth=1.6)
    if right.get_legend_handles_labels()[0]:
        right.legend(frameon=False, fontsize="small")
    right.set_title("histograms", loc="left")
    right.set_xlabel("value")
    right.set_ylabel("rows")
    dress(right)
    parts.append(embedded(figure, f"{pair['pred']} against {pair['target']} over "
                                  f"{plural(found.get('count') or len(points), 'row')}, and their histograms"))
    return "\n".join(parts)


def confusion_axis(axis, pair, classes, confusion):
    count = len(classes)
    axis.imshow(confusion, cmap="Blues")
    axis.set_xticks(range(count), classes, rotation=90 if count > 8 else 0)
    axis.set_yticks(range(count), classes)
    axis.set_xlabel(f"predicted ({pair['pred']})")
    axis.set_ylabel(f"truth ({pair['target']})")
    axis.set_title("confusion matrix", loc="left")
    if count > ANNOTATED:
        return
    peak = max((max(row) for row in confusion if row), default=0) or 1
    for row_position, row in enumerate(confusion):
        for column_position, value in enumerate(row):
            axis.text(column_position, row_position, str(value), ha="center", va="center", fontsize=8,
                      color="white" if value > peak / 2 else INK)


def roc_curves(found):
    roc = found.get("roc")
    if not isinstance(roc, dict) or not roc:
        return []
    if "fpr" in roc:
        return [(None, roc)]
    return [(str(label), curve) for label, curve in roc.items() if isinstance(curve, dict)]


def roc_axis(axis, curves):
    for position, (label, roc) in enumerate(curves):
        name = None if label is None else f"{label}, auc {shown(roc.get('auc'))}"
        axis.plot(roc.get("fpr") or [], roc.get("tpr") or [], color=color_of(position), linewidth=1.6, label=name)
    axis.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1)
    if curves[0][0] is None:
        axis.set_title(f"roc curve, auc {shown(curves[0][1].get('auc'))}", loc="left")
    else:
        axis.set_title("roc, one class against the rest", loc="left")
        if len(curves) <= LEGEND_LIMIT:
            axis.legend(frameon=False, fontsize="small")
    axis.set_xlabel("false positive rate")
    axis.set_ylabel("true positive rate")
    dress(axis)


def classification_numbers(pair, found, curves):
    if pair.get("task") != "binary":
        keys = ("count", "accuracy", "macro_f1", "weighted_f1")
        return [table([key.replace("_", " ") for key in keys], [[found.get(key) for key in keys]])]
    keys = ("count", "accuracy", "precision", "recall", "f1", "balanced_accuracy")
    auc = [curves[0][1].get("auc")] if curves else []
    parts = [table([key.replace("_", " ") for key in keys] + ["roc auc"] * len(auc),
                   [[found.get(key) for key in keys] + auc])]
    if found.get("score"):
        rule = THRESHOLD_RULES.get(found.get("threshold_rule"), found.get("threshold_rule"))
        text = f"{found['score']} >= {shown(found.get('threshold'))} reads as {found.get('positive')}, {rule}"
    else:
        text = f"no score column, so the predicted labels are counted; the positive class is {found.get('positive')}"
    return parts + [paragraph(text, "muted")]


def classification_part(pair, found):
    curves = roc_curves(found)
    parts = classification_numbers(pair, found, curves)
    per_class = found.get("per_class") or {}
    if per_class:
        keys = ("precision", "recall", "f1", "support")
        aucs = {label: curve.get("auc") for label, curve in curves if label is not None}
        parts.append(table(["class", *keys, *(["roc auc"] if aucs else [])],
                           [[label, *(scores.get(key) for key in keys), *([aucs.get(str(label))] if aucs else [])]
                            for label, scores in per_class.items()]))
    classes = [str(label) for label in found.get("classes") or []]
    confusion = found.get("confusion") or []
    if not found.get("count") or not classes:
        parts.append(paragraph("confusion matrix not drawn: no rows", "muted"))
        return "\n".join(parts)
    if len(classes) > MAX_CLASSES:
        parts.append(paragraph(f"confusion matrix not drawn: {len(classes)} classes, more than {MAX_CLASSES}",
                               "muted"))
        return "\n".join(parts)
    figure, axes = new_figure(9.6 if curves else 5.2, 4.2, columns=2 if curves else 1)
    confusion_axis(axes[0] if curves else axes, pair, classes, confusion)
    if curves:
        roc_axis(axes[1], curves)
    caption = f"confusion matrix of {pair['pred']} against {pair['target']}"
    ending = ", and the roc curve" if curves and curves[0][0] is None else ", and the roc curves" if curves else ""
    parts.append(embedded(figure, caption + ending))
    return "\n".join(parts)


def stairs_of(axis, found, title, label):
    counts, edges = (found or {}).get("counts") or [], (found or {}).get("edges") or []
    if counts and len(edges) == len(counts) + 1:
        axis.stairs(counts, edges, color=color_of(0), linewidth=1.6)
    axis.set_title(title, loc="left")
    axis.set_xlabel(label)
    axis.set_ylabel("rows")
    dress(axis)


def embedding_part(pair, found):
    pca = found.get("pca") or {}
    explained = list(pca.get("explained") or [None, None])[:2]
    parts = [table(["count", "dims", "first component", "second component"],
                   [[found.get("count"), found.get("dims"), *explained]])]
    colors = pca.get("color") or [None] * len(pca.get("x") or [])
    points = [(x, y, str(color)) for x, y, color in zip(pca.get("x") or [], pca.get("y") or [], colors)
              if finite(x) and finite(y)]
    if not points:
        parts.append(paragraph("no finite rows to draw", "muted"))
        return "\n".join(parts)
    figure, (left, right) = new_figure(9.6, 3.8, columns=2)
    names = sorted({color for _, _, color in points}) if pca.get("color") else []
    grouped = 0 < len(names) <= LEGEND_LIMIT
    for index, name in enumerate(names if grouped else [None]):
        chosen = [(x, y) for x, y, color in points if name is None or color == name]
        left.scatter(*zip(*chosen), s=10, alpha=0.7, color=color_of(index), linewidths=0, label=name)
    if grouped:
        left.legend(frameon=False, fontsize="small", title=pca.get("color_by"))
    left.set_title("pca of the rows", loc="left")
    left.set_xlabel("first component")
    left.set_ylabel("second component")
    dress(left)
    stairs_of(right, found.get("norm_hist"), "vector length", "length")
    colored = f", colored by {pca.get('color_by')}" if grouped else ""
    parts.append(embedded(figure, f"pca map of {pair['pred']} over {plural(len(points), 'row')}{colored}, and the "
                                  f"length of each vector"))
    return "\n".join(parts)


def values_part(pair, found):
    labels = found.get("labels")
    if isinstance(labels, dict):
        rows = list(zip(labels.get("labels") or [], labels.get("counts") or []))
        return "\n".join([table(["count"], [[found.get("count")]]), table(["label", "rows"], rows)])
    quantiles = found.get("quantiles") or {}
    parts = [table(["count", "mean", "std", *quantiles],
                   [[found.get("count"), found.get("mean"), found.get("std"), *quantiles.values()]])]
    if (found.get("hist") or {}).get("counts"):
        figure, axis = new_figure(5.2, 3.4)
        stairs_of(axis, found["hist"], "distribution", pair["pred"])
        parts.append(embedded(figure, f"distribution of {pair['pred']} over {plural(found.get('count') or 0, 'row')}"))
    return "\n".join(parts)


def pair_heading(pair):
    task = pair.get("task") or pair["kind"]
    named = pair["kind"] if task == pair["kind"] else f"{pair['kind']} ({task})"
    if pair.get("target") is not None:
        return f"{pair['target']} against {pair['pred']}, {named}"
    columns = pair.get("columns") or [pair["pred"]]
    return f"{pair['pred']}{f', {len(columns)} columns' if len(columns) > 1 else ''}, {named}"


def predictions_part(directory):
    if not (Path(directory) / PREDICTIONS).is_file():
        return ""
    try:
        found = prediction_stats(directory, reader.read_plan(directory))
    except (KalfaError, OSError, ValueError) as exc:
        return paragraph(f"{PREDICTIONS} could not be read: {exc}", "note")
    if found is None:
        return paragraph(f"{PREDICTIONS} could not be read", "note")
    parts = ["<h4>predictions</h4>", paragraph(f"{plural(found.get('rows') or 0, 'row')} in {PREDICTIONS}", "muted")]
    pairs = found.get("pairs") or []
    if not pairs:
        parts.append(paragraph("the table has no pred_ column", "note"))
    drawers = {"classification": classification_part, "embedding": embedding_part, "values": values_part}
    for pair in pairs:
        parts.append(f"<h5>{escape(pair_heading(pair))}</h5>")
        parts.append(drawers.get(pair["kind"], regression_part)(pair, pair.get("stats") or {}))
    return "\n".join(parts)
