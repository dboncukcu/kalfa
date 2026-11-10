import numpy
import pandas

from .measures import histogram
from .pairs import label_columns
from .tables import finite, labels, numbers_of, ordered_labels, plain, present, texts


RATES = ("accuracy", "precision", "recall", "f1", "specificity", "balanced_accuracy")
SWEEP = 101
CURVE = 2000
COMPLEMENT = "1 - "
NEGATED = "-"


def ratio(part, whole):
    return float(part) / float(whole) if whole else 0.0


def positive_column(scores, output, classes):
    named = label_columns(scores, output, classes).get(classes[1])
    if named is not None:
        return named
    if len(scores) == 2:
        return scores[1]
    return scores[0] if len(scores) == 1 else None


def thin(*arrays, limit=CURVE):
    size = len(arrays[0])
    if size <= limit:
        return arrays
    chosen = numpy.unique(numpy.round(numpy.linspace(0, size - 1, limit)).astype(int))
    return tuple(array[chosen] for array in arrays)


def binary_curves(truth, probabilities):
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

    if len(numpy.unique(truth)) < 2:
        return None, None
    fpr, tpr, _ = roc_curve(truth, probabilities)
    precision, recall, _ = precision_recall_curve(truth, probabilities)
    fpr, tpr = thin(fpr, tpr)
    precision, recall = thin(precision, recall)
    roc = {"fpr": plain(fpr), "tpr": plain(tpr), "auc": float(roc_auc_score(truth, probabilities))}
    pr = {"precision": plain(precision), "recall": plain(recall),
          "ap": float(average_precision_score(truth, probabilities))}
    return roc, pr


def per_class_of(classes, confusion):
    found = {}
    for index, label in enumerate(classes):
        hits = int(confusion[index, index])
        predicted, actual = int(confusion[:, index].sum()), int(confusion[index, :].sum())
        precision, recall = ratio(hits, predicted), ratio(hits, actual)
        found[label] = {"precision": precision, "recall": recall,
                        "f1": ratio(2 * precision * recall, precision + recall), "support": actual}
    return found


def confusion_of(classes, truth, guess):
    position = {label: index for index, label in enumerate(classes)}
    confusion = numpy.zeros((len(classes), len(classes)), dtype=numpy.int64)
    for actual, predicted in zip(truth, guess):
        confusion[position[actual], position[predicted]] += 1
    return confusion


def normalized(confusion):
    totals = confusion.sum(axis=1, keepdims=True)
    shares = numpy.divide(confusion, totals, out=numpy.zeros(confusion.shape, dtype=numpy.float64),
                          where=totals > 0)
    return shares.tolist()


def classification_stats(frame, pred, target, raw=None, scores=None, classes=None):
    guess, truth = texts(frame[pred]), texts(frame[target])
    keep = present(guess) & present(truth)
    guess, truth = guess[keep], truth[keep]
    classes = labels(classes) or []
    classes = classes + ordered_labels((set(truth.tolist()) | set(guess.tolist())) - set(classes))
    confusion = confusion_of(classes, truth, guess)
    count = int(len(truth))
    roc = pr = None
    output = pred[len("pred_"):] if pred.startswith("pred_") else pred
    column = positive_column(list(scores), output, classes) if scores and len(classes) == 2 else None
    if column is not None and column in frame.columns and count:
        probabilities = pandas.to_numeric(frame[column], errors="coerce").to_numpy(dtype=numpy.float64)[keep]
        finite_rows = numpy.isfinite(probabilities)
        roc, pr = binary_curves((truth[finite_rows] == classes[1]).astype(int), probabilities[finite_rows])
    return {"count": count, "accuracy": float(numpy.trace(confusion) / count) if count else None,
            "classes": classes, "confusion": confusion.tolist(), "per_class": per_class_of(classes, confusion),
            "roc": roc, "pr": pr}


def rates(tn, fp, fn, tp):
    count = tn + fp + fn + tp
    if not count:
        return {key: None for key in RATES}
    recall, specificity = ratio(tp, tp + fn), ratio(tn, tn + fp)
    present_rates = [rate for rate, size in ((recall, tp + fn), (specificity, tn + fp)) if size]
    return {"accuracy": ratio(tp + tn, count), "precision": ratio(tp, tp + fp), "recall": recall,
            "f1": ratio(2 * tp, 2 * tp + fp + fn), "specificity": specificity,
            "balanced_accuracy": float(sum(present_rates) / len(present_rates))}


def in_unit(values):
    found = finite(values)
    return not len(found) or (float(found.min()) >= 0.0 and float(found.max()) <= 1.0)


def best_threshold(actual, values):
    if not len(values):
        return None
    order = numpy.argsort(-values, kind="stable")
    ranked, hits = values[order], actual[order]
    tp, fp = numpy.cumsum(hits), numpy.cumsum(~hits)
    ends = numpy.r_[ranked[1:] != ranked[:-1], True]
    tp, fp, ranked = tp[ends], fp[ends], ranked[ends]
    f1 = 2.0 * tp / numpy.maximum(tp + fp + int(hits.sum()), 1)
    return float(ranked[int(numpy.argmax(f1))])


def sweep_of(actual, values, unit):
    if unit:
        grid = numpy.linspace(0.0, 1.0, SWEEP)
    else:
        grid = numpy.unique(numpy.quantile(values, numpy.linspace(0.0, 1.0, SWEEP)))
    positives, negatives = numpy.sort(values[actual]), numpy.sort(values[~actual])
    tp = len(positives) - numpy.searchsorted(positives, grid, side="left")
    fp = len(negatives) - numpy.searchsorted(negatives, grid, side="left")
    fn = len(positives) - tp
    return {"threshold": [float(value) for value in grid],
            "precision": [ratio(hit, hit + miss) for hit, miss in zip(tp, fp)],
            "recall": [ratio(hit, len(positives)) for hit in tp],
            "f1": [ratio(2 * hit, 2 * hit + miss + lost) for hit, miss, lost in zip(tp, fp, fn)]}


def score_hist_of(actual, values, unit, bins):
    if unit:
        edges = numpy.histogram_bin_edges(values, bins=bins, range=(0.0, 1.0))
    else:
        edges = numpy.histogram_bin_edges(values, bins=bins)
    return {"edges": [float(edge) for edge in edges],
            "negative": histogram(values[~actual], edges=edges)["counts"],
            "positive": histogram(values[actual], edges=edges)["counts"]}


def compared(values, threshold):
    if threshold is None:
        return numpy.zeros(len(values), dtype=bool)
    with numpy.errstate(invalid="ignore"):
        return values >= threshold


def score_values(frame, score):
    if score in frame.columns:
        return numbers_of(frame[score])
    if score.startswith(COMPLEMENT):
        return 1.0 - numbers_of(frame[score[len(COMPLEMENT):]])
    if score.startswith(NEGATED) and score[len(NEGATED):] in frame.columns:
        return -numbers_of(frame[score[len(NEGATED):]])
    return numbers_of(frame[score])


def binary_rows(frame, pred, target, labels, score, threshold):
    negative, positive = labels
    truth = texts(frame[target])
    keep = present(truth)
    if score is None:
        guess = texts(frame[pred])
        return truth, guess, None, keep & present(guess)
    values = score_values(frame, score)
    guess = numpy.where(compared(values, threshold), positive, negative).astype(object)
    return truth, guess, values, keep & numpy.isfinite(values)


def binary_stats(frame, pred, target, labels, score=None, threshold=None, rule=None, unit=True, bins=40,
                 candidates=None):
    positive = labels[1]
    truth, guess, values, keep = binary_rows(frame, pred, target, labels, score, threshold)
    actual, predicted = (truth == positive)[keep].astype(bool), (guess == positive)[keep].astype(bool)
    count = int(keep.sum())
    tp, fp = int((actual & predicted).sum()), int((~actual & predicted).sum())
    fn = int((actual & ~predicted).sum())
    tn = count - tp - fp - fn
    confusion = numpy.asarray([[tn, fp], [fn, tp]], dtype=numpy.int64)
    stats = {"count": count, "labels": list(labels), "positive": positive, "score": score, "threshold": threshold,
             "threshold_rule": rule, "confusion": confusion.tolist(), "confusion_normalized": normalized(confusion),
             **rates(tn, fp, fn, tp), "roc": None, "pr": None, "score_hist": None, "sweep": None,
             "candidates": list(candidates or []), "classes": list(labels),
             "per_class": per_class_of(list(labels), confusion)}
    if values is None or not count:
        return stats
    scored = values[keep]
    stats["roc"], stats["pr"] = binary_curves(actual.astype(int), scored)
    stats["score_hist"] = score_hist_of(actual, scored, unit, bins)
    stats["sweep"] = sweep_of(actual, scored, unit)
    return stats


def multiclass_rows(frame, pred, target):
    truth, guess = texts(frame[target]), texts(frame[pred])
    return truth, guess, present(truth) & present(guess)


def multiclass_stats(frame, pred, target, classes=None, scores=None):
    truth, guess, keep = multiclass_rows(frame, pred, target)
    truth, guess = truth[keep], guess[keep]
    classes = list(classes or [])
    classes = classes + ordered_labels((set(truth.tolist()) | set(guess.tolist())) - set(classes))
    confusion = confusion_of(classes, truth, guess)
    count = int(len(truth))
    per_class = per_class_of(classes, confusion)
    seen = [label for index, label in enumerate(classes) if confusion[index, :].sum() or confusion[:, index].sum()]
    output = pred[len("pred_"):] if pred.startswith("pred_") else pred
    columns = label_columns(scores or [], output, classes)
    curves = {}
    for label in classes:
        column = columns.get(label)
        if column is None or column not in frame.columns or not count:
            continue
        probabilities = numbers_of(frame[column])[keep]
        rows = numpy.isfinite(probabilities)
        roc, _ = binary_curves((truth[rows] == label).astype(int), probabilities[rows])
        if roc is not None:
            curves[label] = roc
    return {"count": count, "classes": classes, "confusion": confusion.tolist(),
            "confusion_normalized": normalized(confusion), "per_class": per_class,
            "accuracy": ratio(numpy.trace(confusion), count) if count else None,
            "macro_f1": float(numpy.mean([per_class[label]["f1"] for label in seen])) if seen else None,
            "weighted_f1": float(sum(item["f1"] * item["support"] for item in per_class.values()) / count)
            if count else None,
            "roc": curves or None, "pr": None}
