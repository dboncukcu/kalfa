"""Plots: partial legos called after training with the predictions, the history and the report models."""

import numpy

from ..registration import lego
from . import figure
from .log import logger_for

logger = logger_for("after.plots")

SKIPPED = ("turn", "global_step", "rules")


def uri_of(plot):
    """The registry URI of a built plot (a function or a partial of one), None when it is not registered."""
    from cirak.registry import registry

    base = getattr(plot, "func", plot)
    for uri in registry.uris():
        entry = registry.lookup(uri)
        if entry is not None and entry.target is base:
            return uri
    return None


def series_of(history, series=None):
    """History columns as name -> list of values; ``series`` restricts them."""
    found = {}
    for line in history or []:
        for key, value in line.items():
            if key in SKIPPED or key.startswith("lr/") or not isinstance(value, (int, float)):
                continue
            found.setdefault(key, []).append(value)
    if series:
        missing = [name for name in series if name not in found]
        if missing:
            raise ValueError(f"loss_curve: {missing} are not in the history; the series are {sorted(found)}")
        found = {name: found[name] for name in series}
    return found


@lego("/plot/kalfa/loss_curve", partial=True, alias="loss_curve",
            description="Every history series over the turns, or the named ones")
def loss_curve(predictions, history, models, record, series=None, log=False, name=None):
    found = series_of(history, series)
    if not found:
        return None
    drawing, axis = figure.single(width=8.0, height=5.0)
    for label, values in found.items():
        axis.plot(range(1, len(values) + 1), values, label=label)
    if log:
        axis.set_yscale("log")
    axis.legend(loc="upper right", ncols=1 if len(found) < 6 else 2)
    last = ", ".join(f"{label} {values[-1]:.4g}" for label, values in list(found.items())[:4])
    figure.label(axis, "Training history", "turn", "value", note=f"last turn: {last}" if last else None)
    figure.save(drawing, record, name or "loss_curve")
    return None


def true_column(pred, targets):
    """The observed column a ``pred_<wire>_<field>`` column belongs to: the longest target name it ends with."""
    matches = [target for target in targets if pred == f"pred_{target}" or pred.endswith(f"_{target}")]
    return max(matches, key=len) if matches else None


def panel_title(pred, target, paired):
    if paired.count(target) < 2 or pred == f"pred_{target}" or not pred.endswith(f"_{target}"):
        return target
    return f"{target} ({pred[len('pred_'):-len(target) - 1]})"


def prediction_pairs(predictions):
    """The (prediction column, target column) pairs of a predictions frame, numeric ones only."""
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


@lego("/plot/kalfa/pred_vs_true", partial=True, alias="pred_vs_true",
            description="Predicted against true values of the test set, one panel per predicted field with its R2, "
                        "as a hexbin density over many points and a scatter over few; the panel is titled with the "
                        "field name, plus the output wire when two outputs predict the same field")
def pred_vs_true(predictions, history, models, record, name=None, columns=4, kind="auto", gridsize=70):
    pairs = prediction_pairs(predictions)
    if not pairs:
        return None
    width = max(1, min(int(columns or 4), len(pairs)))
    rows = -(-len(pairs) // width)
    drawing, axes = figure.grid(rows, width, width=5.4, height=4.6)
    panels = [axis for row in axes for axis in row]
    paired = [field for _, field in pairs]
    for axis, (pred, target) in zip(panels, pairs):
        true, guess = figure.finite(predictions[target].to_numpy(), predictions[pred].to_numpy())
        if not len(true):
            axis.axis("off")
            continue
        dense = kind == "hexbin" or (kind == "auto" and len(true) >= 2000)
        if dense:
            figure.density(drawing, axis, true, guess, gridsize)
        else:
            axis.scatter(true, guess, s=12, alpha=0.5, color=figure.CATEGORICAL[0], edgecolors="none")
        low = float(min(true.min(), guess.min()))
        high = float(max(true.max(), guess.max()))
        axis.plot([low, high], [low, high], color=figure.CATEGORICAL[1], linewidth=2, label="perfect")
        axis.legend(loc="upper left")
        figure.label(axis, panel_title(pred, target, paired), "true", "predicted",
                     note=f"R2 = {r2_of(true, guess):.4f} on {len(true):,} points")
    for axis in panels[len(pairs):]:
        axis.axis("off")
    figure.save(drawing, record, name or "pred_vs_true")
    return None


def plot_inputs(plot, inputs, predictions, history, models):
    """The extra inputs of a plot resolved by the types its refs fact declares: history keys, fields, models."""
    from cirak.registry import registry

    uri = uri_of(plot)
    refs = registry.facts(uri).refs if isinstance(uri, str) else {}
    resolved = {}
    for param, name in (inputs or {}).items():
        ref_type = refs.get(param)
        if ref_type == "history":
            resolved[param] = [line.get(name) for line in history or []]
        elif ref_type == "field":
            resolved[param] = predictions[name] if predictions is not None and name in predictions else None
        elif ref_type == "model":
            resolved[param] = models.get(name) if models else None
        else:
            resolved[param] = name
    return resolved


def _accepts(plot, name):
    import inspect

    try:
        parameters = inspect.signature(plot).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters or any(parameter.kind is parameter.VAR_KEYWORD for parameter in parameters.values())


@lego("/lego/kalfa/run_all", returns=None,
            bus=["record", "composites", "valid_loader", "test_loader"],
            description="Run every plot of the plots table with the predictions, the history and the models; keys "
                        "carry the extra inputs a plot names; plots that take loaders, predicts or name get them, "
                        "name being the definition key the file is named after; figures carries the figure "
                        "settings of the config")
def run_all(predictions, history, models, plots, keys=None, predicts=None, figures=None, composites=None,
            valid_loader=None, test_loader=None, record=None):
    keys = keys or {}
    figure.configure(figures)
    everything = {**dict(composites or {}), **dict(models or {})}
    loaders = {"valid": valid_loader, "test": test_loader}
    for name, plot in (plots or {}).items():
        logger.debug(f"drawing {name}")
        definition = keys.get(name) or {}
        extra = plot_inputs(plot, definition.get("inputs"), predictions, history, everything)
        if _accepts(plot, "loaders"):
            extra["loaders"] = loaders
        if _accepts(plot, "predicts"):
            extra["predicts"] = predicts
        if _accepts(plot, "sets"):
            extra["sets"] = definition.get("sets")
        if _accepts(plot, "name"):
            extra["name"] = name
        size = {key: definition[key] for key in ("width", "height") if definition.get(key) is not None}
        if size:
            figure.configure({**figure.settings(), **size})
        try:
            plot(predictions=predictions, history=history, models=everything, record=record, **extra)
        finally:
            if size:
                figure.configure(figures)
    if plots:
        logger.info(f"plots: {', '.join(plots)}")
    return None


def _report_loader(loaders, set_name=None):
    """The loader image plots draw from: the named set, else test when it has items, else valid."""
    from .feed import sized

    names = [set_name] if set_name else ["test", "valid"]
    for name in names:
        loader = (loaders or {}).get(name)
        if loader is not None and sized(loader.dataset):
            return name, loader
    return None, None


def _image_grid(axis, tensor):
    array = tensor.detach().cpu().float().numpy()
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = array.transpose(1, 2, 0)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    low, high = float(array.min()), float(array.max())
    if high > low:
        array = (array - low) / (high - low)
    axis.imshow(numpy.clip(array, 0.0, 1.0), cmap="gray" if array.ndim == 2 else None)
    axis.grid(visible=False)
    axis.set_xticks([])
    axis.set_yticks([])


@lego("/plot/kalfa/image_grid", partial=True, alias="image_grid",
            description="n outputs of the predicts model on the report set as an image grid")
def image_grid(predictions, history, models, record, loaders=None, predicts=None, n=16, set=None, name=None):
    import torch

    from .runtime import call_model, named_outputs, resolve_model

    set_name, loader = _report_loader(loaders, set)
    if loader is None or predicts is None:
        return None
    model = resolve_model(predicts, models)
    model.eval()
    count = min(int(n), len(loader.dataset))
    if count == 0:
        return None
    items = [loader.dataset[position] for position in range(count)]
    batch = {key: torch.stack([item[key] for item in items]) for key in items[0]}
    device = next(iter(model.parameters()), torch.zeros(1)).device
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.no_grad():
        outputs = named_outputs(model, call_model(model, batch))
    images = outputs[next(iter(outputs))]
    if images.ndim != 4:
        return None
    columns = min(8, count)
    rows = (count + columns - 1) // columns
    drawing, axes = figure.tiles(rows, columns)
    for position in range(rows * columns):
        axis = axes[position // columns][position % columns]
        if position < count:
            _image_grid(axis, images[position])
        else:
            axis.axis("off")
    axes[0][0].set_ylabel(set_name)
    figure.save(drawing, record, name or "image_grid")
    return None


@lego("/plot/kalfa/image_pairs", partial=True, alias="image_pairs",
            description="n inputs of the report set next to the predicts model's outputs (reconstructions)")
def image_pairs(predictions, history, models, record, loaders=None, predicts=None, n=8, set=None, name=None):
    import torch

    from .runtime import call_model, named_outputs, resolve_model

    set_name, loader = _report_loader(loaders, set)
    if loader is None or predicts is None:
        return None
    model = resolve_model(predicts, models)
    model.eval()
    count = min(int(n), len(loader.dataset))
    items = [loader.dataset[position] for position in range(count)]
    batch = {key: torch.stack([item[key] for item in items]) for key in items[0]}
    device = next(iter(model.parameters()), torch.zeros(1)).device
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.no_grad():
        outputs = named_outputs(model, call_model(model, batch))
    wire = next(iter(outputs))
    inputs = batch[model.inputs[0]]
    drawing, axes = figure.sized(1.6 * count, 3.4, 2, count)
    for position in range(count):
        _image_grid(axes[0][position], inputs[position])
        _image_grid(axes[1][position], outputs[wire][position])
    axes[0][0].set_ylabel(set_name)
    axes[1][0].set_ylabel(wire)
    figure.save(drawing, record, name or "image_pairs")
    return None


def _scores_and_labels(predictions):
    preds = [column for column in predictions.columns if column.startswith("raw_")]
    targets = [column for column in predictions.columns
               if not column.startswith(("pred_", "raw_")) and column != "row"]
    if not preds or not targets:
        return None, None
    return predictions[preds[0]].to_numpy(dtype="float64"), predictions[targets[0]].to_numpy()


@lego("/plot/kalfa/class_histogram", partial=True, alias="class_histogram",
            description="Histogram of the raw scores of the test set, one series per target class")
def class_histogram(predictions, history, models, record, bins=40, name=None):
    scores, labels = _scores_and_labels(predictions)
    if scores is None:
        return None
    drawing, axis = figure.single(width=8.0, height=5.0)
    classes = sorted(set(labels.tolist()))
    for position, label in enumerate(classes):
        axis.hist(scores[labels == label], bins=bins, alpha=0.55, edgecolor="none",
                  color=figure.CATEGORICAL[position % len(figure.CATEGORICAL)], label=str(label))
    axis.legend(loc="upper right")
    figure.label(axis, "Score by class", "score", "points",
                 note=f"{len(scores):,} test points over {len(classes)} classes")
    figure.save(drawing, record, name or "class_histogram")
    return None


def _area(x, y):
    return float(abs(numpy.sum(numpy.diff(x) * (y[:-1] + y[1:]) / 2.0)))


def _binary_curve(predictions, record, kind, xlabel, ylabel, name=None):
    import torch
    from torchmetrics.functional.classification import binary_precision_recall_curve, binary_roc

    scores, labels = _scores_and_labels(predictions)
    if scores is None or len(set(labels.tolist())) < 2:
        return None
    score = torch.as_tensor(numpy.array(scores, dtype="float32"))
    label = torch.as_tensor(numpy.array(labels)).long()
    if kind == "binary_roc":
        x, y, _ = binary_roc(score, label)
        title, note = "ROC", f"AUC = {_area(x.numpy(), y.numpy()):.4f}"
    else:
        precision, recall, _ = binary_precision_recall_curve(score, label)
        x, y = recall, precision
        title, note = "Precision and recall", f"AP = {_area(x.numpy(), y.numpy()):.4f}"
    drawing, axis = figure.single(width=6.0, height=5.0)
    axis.plot(x.numpy(), y.numpy(), color=figure.CATEGORICAL[0], label=note)
    if kind == "binary_roc":
        axis.plot([0, 1], [0, 1], color=figure.INK_MUTED, linewidth=1, linestyle="--", label="chance")
    else:
        axis.axhline(float(label.float().mean()), color=figure.INK_MUTED, linewidth=1, linestyle="--",
                     label="base rate")
    axis.legend(loc="lower right" if kind == "binary_roc" else "lower left")
    figure.label(axis, title, xlabel, ylabel, note=f"{len(scores):,} test points")
    figure.save(drawing, record, name or kind)
    return None


@lego("/plot/torchmetrics/binary_roc", partial=True,
            description="ROC curve of the raw test scores against the binary target")
def binary_roc(predictions, history, models, record, name=None):
    return _binary_curve(predictions, record, "binary_roc", "false positive rate", "true positive rate", name)


@lego("/plot/torchmetrics/binary_precision_recall_curve", partial=True,
            description="Precision recall curve of the raw test scores against the binary target")
def binary_precision_recall_curve(predictions, history, models, record, name=None):
    return _binary_curve(predictions, record, "binary_precision_recall_curve", "recall", "precision", name)


def _draw_models(models, loaders, record, stem):
    import warnings

    import torch

    try:
        from torchview import draw_graph
    except ImportError:
        logger.warning("architecture: torchview is not installed, so the models are written as text only; "
                       "pip install torchview (and the graphviz dot binary) for the drawing")
        return

    loader = _report_loader(loaders)[1]
    batch = next(iter(loader), None) if loader is not None else None
    if batch is None:
        logger.debug("architecture: no batch to trace with, the text is all there is")
        return
    for label, model in (models or {}).items():
        wires = getattr(model, "inputs", None)
        if not wires or any(wire not in batch for wire in wires):
            continue
        device = next(iter(model.parameters()), torch.zeros(1)).device
        try:
            drawing = draw_graph(model, input_data=[batch[wire][:2].to(device) for wire in wires],
                                 graph_name=label, expand_nested=True)
            drawing.visual_graph.render(str(figure.target(record, f"{stem}_{label}")), format="png", cleanup=True)
            logger.debug(f"architecture: drew {label}")
        except Exception as exc:
            warnings.warn(f"architecture: torchview could not draw {label}: {type(exc).__name__}: {exc}")


@lego("/plot/kalfa/architecture", partial=True, alias="architecture",
            description="The report models printed as text under plots/architecture.txt, and drawn under "
                        "plots/architecture_<model>.png when torchview and graphviz are installed")
def architecture(predictions, history, models, record, loaders=None, name=None):
    lines = []
    for label, model in (models or {}).items():
        lines.append(f"== {label}")
        lines.append(repr(model))
        lines.append("")
    stem = name or "architecture"
    figure.target(record, f"{stem}.txt").write_text("\n".join(lines))
    _draw_models(models, loaders, record, stem)
    return None


@lego("/plot/kalfa/confusion_matrix", partial=True, alias="confusion_matrix",
            description="Confusion matrix of the decoded test predictions against the target labels, counts and "
                        "row shares in every cell")
def confusion_matrix(predictions, history, models, record, name=None):
    if predictions is None or len(predictions) == 0:
        return None
    preds = [column for column in predictions.columns if column.startswith("pred_")]
    targets = [column for column in predictions.columns
               if not column.startswith(("pred_", "raw_")) and column != "row"]
    if not preds or not targets:
        return None
    from sklearn.metrics import confusion_matrix as sk_confusion

    truth = predictions[targets[0]].astype(str).to_numpy()
    guess = predictions[preds[0]].astype(str).to_numpy()
    labels = sorted(set(truth.tolist()) | set(guess.tolist()))
    matrix = sk_confusion(truth, guess, labels=labels)
    shares = matrix / numpy.maximum(matrix.sum(axis=1, keepdims=True), 1)
    side = 1.8 + 0.8 * len(labels)
    drawing, axes = figure.sized(side, side * 0.86)
    axis = axes[0][0]
    drawn = axis.imshow(shares, cmap=figure.sequential(), vmin=0, vmax=1)
    for row in range(len(labels)):
        for column in range(len(labels)):
            axis.text(column, row, f"{matrix[row, column]:,}\n{shares[row, column] * 100:.1f}%",
                      ha="center", va="center", fontsize=9,
                      color=figure.INK if shares[row, column] < 0.55 else "#ffffff")
    axis.set_xticks(range(len(labels)), labels)
    axis.set_yticks(range(len(labels)), labels)
    axis.grid(visible=False)
    figure.colorbar(drawing, drawn, axis, "row share", fraction=0.045)
    figure.label(axis, "Confusion matrix", "predicted", "true", note=f"{len(truth):,} test points")
    figure.save(drawing, record, name or "confusion_matrix")
    return None


@lego("/plot/kalfa/forecast_samples", partial=True, alias="forecast_samples",
            description="n sample windows of the test set: the true horizon against the predicted one")
def forecast_samples(predictions, history, models, record, n=6, name=None):
    if predictions is None or len(predictions) == 0:
        return None
    preds = sorted([column for column in predictions.columns if column.startswith("pred_")],
                   key=lambda column: int(column.rsplit("_", 1)[1]) if column.rsplit("_", 1)[1].isdigit() else 0)
    truths = [column for column in predictions.columns
              if not column.startswith(("pred_", "raw_")) and column != "row"]
    if not preds or not truths:
        return None
    count = min(int(n), len(predictions))
    picked = [int(round(position)) for position in numpy.linspace(0, len(predictions) - 1, count)]
    drawing, axes = figure.sized(figure.width_of(8.0), 2.2 * count, count, 1)
    for axis, position in zip(axes[:, 0], picked):
        row = predictions.iloc[position]
        axis.plot([row[column] for column in truths], color=figure.CATEGORICAL[0], label="true")
        axis.plot([row[column] for column in preds], color=figure.CATEGORICAL[1], linestyle="--",
                  label="predicted")
        figure.label(axis, f"row {row['row']}", None, None)
    axes[0, 0].legend(loc="upper right")
    figure.save(drawing, record, name or "forecast_samples")
    return None


def _turn_files(record, suffix):
    from pathlib import Path

    return sorted((Path(record) / "samples").glob(f"turn_*{suffix}"))


@lego("/plot/kalfa/samples_gif", partial=True, alias="samples_gif",
            description="The per turn sample grids of samples/turn_*.png as an animation; skipped with a warning "
                        "when there are none")
def samples_gif(predictions, history, models, record, name=None, duration=400):
    import warnings

    from PIL import Image

    frames = _turn_files(record, ".png")
    if not frames:
        warnings.warn("samples_gif: no samples/turn_*.png in the record; add a sample_writer metric")
        return None
    images = []
    for frame in frames:
        with Image.open(frame) as handle:
            images.append(handle.convert("RGB"))
    images[0].save(figure.target(record, f"{name or 'samples_gif'}.gif"), save_all=True,
                   append_images=images[1:], duration=int(duration), loop=0)
    return None


@lego("/plot/kalfa/samples_matrix", partial=True, alias="samples_matrix",
            description="A matrix of the per turn samples of samples/turn_*.pt: one row per turn, n columns; "
                        "skipped with a warning when there are none")
def samples_matrix(predictions, history, models, record, name=None, n=8):
    import warnings

    import torch

    files = _turn_files(record, ".pt")
    if not files:
        warnings.warn("samples_matrix: no samples/turn_*.pt in the record; add a sample_writer metric")
        return None
    rows = []
    for path in files:
        samples = torch.load(path, weights_only=False)
        if isinstance(samples, torch.Tensor) and samples.ndim == 4:
            rows.append((path.stem.split("_", 1)[1].lstrip("0") or "0", samples[:int(n)]))
    if not rows:
        warnings.warn("samples_matrix: the turn samples are not images")
        return None
    columns = max(len(samples) for _, samples in rows)
    drawing, axes = figure.tiles(len(rows), columns)
    for row, (turn, samples) in enumerate(rows):
        for column in range(columns):
            axis = axes[row][column]
            if column < len(samples):
                _image_grid(axis, samples[column])
            else:
                axis.axis("off")
        axes[row][0].set_ylabel(f"turn {turn}")
    figure.save(drawing, record, name or "samples_matrix")
    return None
