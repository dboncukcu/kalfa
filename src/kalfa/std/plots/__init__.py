from kalfa.core.aliases import register
from kalfa.std.plots.classification import (
    MAX_CLASSES,
    SKEW,
    binary_curve,
    binary_precision_recall_curve,
    binary_roc,
    class_histogram,
    classified,
    confusion_matrix,
    floats,
    label_set,
    score_column,
)
from kalfa.std.plots.common import (
    GRID,
    INK,
    LEGEND_LIMIT,
    MUTED,
    color_of,
    directory_of,
    dress,
    label_text,
    quiet,
    spans_decades,
)
from kalfa.std.plots.curves import DEFAULT_PREFIXES, history_series, loss_curve, turn_word
from kalfa.std.plots.images import (
    TEXT_LINES,
    TEXT_WIDTH,
    first_output,
    image_field,
    image_pairs,
    pixels,
    sample_images,
    samples_grid,
    shaped,
    show,
    text_sheet,
)
from kalfa.std.plots.pairs import chosen_pair, pick_pair, pred_column, prediction_pairs, read_predictions
from kalfa.std.plots.predictions import (
    HORIZON,
    forecast_samples,
    forecast_series,
    horizon_pair,
    indexed,
    numeric,
    pred_vs_true,
    r_squared,
)


__all__ = ["DEFAULT_PREFIXES", "GRID", "HORIZON", "INK", "LEGEND_LIMIT", "MAX_CLASSES", "MUTED", "SKEW", "TEXT_LINES",
           "TEXT_WIDTH", "binary_curve", "binary_precision_recall_curve", "binary_roc", "chosen_pair",
           "class_histogram", "classified", "color_of", "confusion_matrix", "directory_of", "dress", "first_output",
           "floats", "forecast_samples", "forecast_series", "history_series", "horizon_pair", "image_field",
           "image_pairs", "indexed", "label_set", "label_text", "loss_curve", "numeric", "pick_pair", "pixels",
           "pred_column", "pred_vs_true", "prediction_pairs", "quiet", "r_squared", "read_predictions",
           "sample_images", "samples_grid", "score_column", "shaped", "show", "spans_decades", "text_sheet",
           "turn_word"]

for _name, _doc in (
    ("loss_curve", "history.jsonl series against the turn; series names the keys, default every train/ and val/ key"),
    ("pred_vs_true", "scatter of each pred_<output> against its target with a y = x line; pairs maps output to target"),
    ("confusion_matrix", "confusion matrix of a label output against its target with counts; pred and target name the "
                         "pair"),
    ("image_pairs", "n test images above their first model output (reconstructions); needs the run"),
    ("samples_grid", "a grid of the first n images in samples/<file>.pt, or the text of samples/<file>.txt"),
    ("forecast_samples", "n windows of a horizon target: the truth and the prediction over the horizon steps"),
    ("class_histogram", "the histogram of the positive class score per true class"),
    ("binary_roc", "the ROC curve of a binary classification with its AUC"),
    ("binary_precision_recall_curve", "the precision recall curve of a binary classification with its AP"),
):
    register(_name, f"kalfa.std.plots.{_name}", slot="plot", doc=_doc)
