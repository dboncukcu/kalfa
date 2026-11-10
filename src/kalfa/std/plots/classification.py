import numpy

from kalfa.core.errors import KalfaError
from kalfa.records import reader, stats
from kalfa.std.plots.common import INK, LEGEND_LIMIT, MUTED, color_of, directory_of, dress, label_text, quiet
from kalfa.std.plots.pairs import chosen_pair, pick_pair, read_predictions


MAX_CLASSES = 64
SKEW = 20.0


def label_set(truth, guess):
    values = set(truth.tolist()) | set(guess.tolist())
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=str)


def confusion_matrix(record, figures, pred=None, target=None):
    directory = directory_of(record)
    frame = read_predictions(directory)
    if frame is None:
        return None
    pair = pick_pair(frame, quiet(reader.read_plan, directory), pred, target)
    if pair is None:
        return None
    column, truth_column = pair
    keep = frame[column].notna() & frame[truth_column].notna()
    guess = frame.loc[keep, column].to_numpy()
    truth = frame.loc[keep, truth_column].to_numpy()
    labels = label_set(truth, guess)
    if not labels or len(labels) > MAX_CLASSES:
        return None
    index = {label: position for position, label in enumerate(labels)}
    matrix = numpy.zeros((len(labels), len(labels)), dtype="int64")
    for true_value, guess_value in zip(truth.tolist(), guess.tolist()):
        matrix[index[true_value], index[guess_value]] += 1
    shares = matrix / numpy.maximum(matrix.sum(axis=1, keepdims=True), 1)
    side = min(16.0, 2.0 + 0.7 * len(labels))
    figure = figures.new(side + 1.2, side)
    axis = figure.subplots()
    drawn = axis.imshow(shares, cmap="Blues", vmin=0.0, vmax=1.0)
    for row in range(len(labels)):
        for col in range(len(labels)):
            axis.text(col, row, f"{matrix[row, col]:,}\n{shares[row, col] * 100:.0f}%", ha="center", va="center",
                      fontsize=8, color="#ffffff" if shares[row, col] > 0.55 else INK)
    names = [label_text(label) for label in labels]
    tilt = len(names) > 6 or max(len(name) for name in names) > 6
    axis.set_xticks(range(len(names)), names, rotation=45 if tilt else 0, ha="right" if tilt else "center")
    axis.set_yticks(range(len(names)), names)
    axis.grid(False)
    figure.colorbar(drawn, ax=axis, fraction=0.046, pad=0.03, label="row share")
    axis.set_title(f"Confusion matrix ({truth_column}, n = {int(matrix.sum()):,})", loc="left")
    axis.set_xlabel(f"predicted ({column[len('pred_'):]})")
    axis.set_ylabel(f"true {truth_column}")
    figures.save(figure, "confusion_matrix")
    return None


def classified(directory, plot, pred, target):
    frame = stats.read_predictions(directory)
    if frame is None or not len(frame):
        return None
    plan = quiet(reader.read_plan, directory)
    fitted = quiet(reader.fields_of, directory)
    pair = chosen_pair(frame, plan, fitted, pred, target, prefer="classification")
    if pair is None:
        return None
    if pair["kind"] != "classification":
        raise KalfaError(f"{plot}: the predictions carry no probabilities for {pair['pred']}; its target "
                         f"{pair['target']} is a regression target, not a label")
    return frame, plan, fitted, pair


def score_column(pair, classes):
    scores = pair["scores"]
    if scores:
        named = stats.positive_column(scores, pair["output"], classes) if classes and len(classes) == 2 else None
        return named or scores[-1]
    return pair["raw"]


def class_histogram(record, figures, pred=None, target=None, bins=40):
    import pandas
    from matplotlib.colors import to_rgba

    directory = directory_of(record)
    found = classified(directory, "class_histogram", pred, target)
    if found is None:
        return None
    frame, plan, fitted, pair = found
    output, truth_column = pair["output"], pair["target"]
    classes = stats.classes_for(plan, fitted, truth_column)
    column = score_column(pair, classes)
    if column is None:
        raise KalfaError(f"class_histogram: the predictions carry no probabilities (prob_{output}_<class>) and no "
                         f"raw_{output} scores for {pair['pred']}")
    scores = pandas.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64")
    known = frame[truth_column].notna().to_numpy()
    keep = numpy.isfinite(scores) & known
    if not keep.any():
        return None
    scores = scores[keep]
    truth = numpy.asarray([label_text(value) for value in frame.loc[keep, truth_column].tolist()], dtype=object)
    seen = sorted({str(value) for value in truth.tolist()})
    order = list(classes or [])
    labels = [label for label in order if label in seen] + [label for label in seen if label not in order]
    if len(labels) > MAX_CLASSES:
        return None
    if column.startswith("prob_") and scores.min() >= 0.0 and scores.max() <= 1.0:
        edges = numpy.linspace(0.0, 1.0, int(bins) + 1)
    else:
        edges = numpy.histogram_bin_edges(scores, bins=int(bins))
    counts = [numpy.histogram(scores[truth == label], bins=edges)[0] for label in labels]
    peaks = [int(count.max()) for count in counts if count.max() > 0]
    log = len(peaks) > 1 and max(peaks) / min(peaks) > SKEW
    figure = figures.new(8.0, 5.0)
    axis = figure.subplots()
    for position, (label, count) in enumerate(zip(labels, counts)):
        color = color_of(figures, position)
        axis.stairs(count, edges, fill=True, baseline=0.5 if log else 0.0, facecolor=to_rgba(color, 0.3),
                    edgecolor=color, linewidth=1.4, label=f"{label} (n = {int(count.sum()):,})")
    if log:
        axis.set_yscale("log")
        axis.set_ylim(bottom=0.5)
    if len(labels) <= LEGEND_LIMIT:
        axis.legend(loc="upper center", frameon=False, title=f"true {truth_column}")
    prefix = f"prob_{output}_"
    axis.set_xlabel(f"P({column[len(prefix):]})" if column.startswith(prefix) else column)
    axis.set_ylabel("rows")
    axis.set_title(f"Score by true class ({truth_column}, n = {len(scores):,})", loc="left")
    dress(axis)
    figures.save(figure, "class_histogram")
    return None


def floats(values):
    return numpy.asarray([numpy.nan if value is None else value for value in values], dtype="float64")


def binary_curve(record, plot, pred, target):
    found = classified(directory_of(record), plot, pred, target)
    if found is None:
        return None
    frame, plan, fitted, pair = found
    output, truth_column = pair["output"], pair["target"]
    if not pair["scores"]:
        raise KalfaError(f"{plot}: the predictions carry no probabilities for {pair['pred']}; a binary curve reads "
                         f"the prob_{output}_<class> columns that a label target writes")
    classes = stats.classes_for(plan, fitted, truth_column)
    result = stats.classification_stats(frame, pair["pred"], truth_column, pair["raw"], pair["scores"], classes)
    if len(result["classes"]) != 2:
        raise KalfaError(f"{plot} needs two classes; {truth_column} has {len(result['classes'])}: "
                         f"{', '.join(result['classes'])}")
    if result["roc"] is None:
        return None
    return result, pair


def binary_roc(record, figures, pred=None, target=None):
    found = binary_curve(record, "binary_roc", pred, target)
    if found is None:
        return None
    result, pair = found
    roc = result["roc"]
    figure = figures.new(5.6, 5.0)
    axis = figure.subplots()
    axis.plot(floats(roc["fpr"]), floats(roc["tpr"]), color=color_of(figures, 0), linewidth=2.0,
              label=f"{pair['output']} (AUC {roc['auc']:.3f})")
    axis.plot([0.0, 1.0], [0.0, 1.0], color=MUTED, linewidth=1.0, linestyle="--", label="chance")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.02)
    axis.legend(loc="lower right", frameon=False)
    positive = result["classes"][1]
    axis.set_title(f"ROC ({pair['target']} = {positive}, AUC {roc['auc']:.3f}, n = {result['count']:,})", loc="left")
    axis.set_xlabel("false positive rate")
    axis.set_ylabel("true positive rate")
    dress(axis)
    figures.save(figure, "binary_roc")
    return None


def binary_precision_recall_curve(record, figures, pred=None, target=None):
    found = binary_curve(record, "binary_precision_recall_curve", pred, target)
    if found is None:
        return None
    result, pair = found
    pr = result["pr"]
    positive = result["classes"][1]
    rate = result["per_class"][positive]["support"] / result["count"]
    figure = figures.new(5.6, 5.0)
    axis = figure.subplots()
    axis.step(floats(pr["recall"]), floats(pr["precision"]), where="post", color=color_of(figures, 0), linewidth=2.0,
              label=f"{pair['output']} (AP {pr['ap']:.3f})")
    axis.axhline(rate, color=MUTED, linewidth=1.0, linestyle="--", label=f"base rate {rate:.3f}")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.02)
    axis.legend(loc="lower left", frameon=False)
    axis.set_title(f"Precision and recall ({pair['target']} = {positive}, AP {pr['ap']:.3f}, "
                   f"n = {result['count']:,})", loc="left")
    axis.set_xlabel("recall")
    axis.set_ylabel("precision")
    dress(axis)
    figures.save(figure, "binary_precision_recall_curve")
    return None
