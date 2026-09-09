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


@lego("/lego/kalfa/run_all", returns=None, bus=["record"],
            description="Run every plot of the plots table with the predictions, the history and the models; keys "
                        "carry the definition level keys (inputs, sets, width, height); bus carries everything else "
                        "the run has (prep, the loaders, the device, the final state) and a plot receives whatever "
                        "its signature names, plus loaders, predicts, sets and name; figures carries the figure "
                        "settings of the config")
def run_all(predictions, history, models, plots, keys=None, predicts=None, figures=None, bus=None, record=None):
    keys = keys or {}
    bus = dict(bus or {})
    before = figure.settings()
    figure.configure(figures)
    everything = {**dict(bus.get("composites") or {}), **dict(models or {})}
    loaders = {name: bus.get(f"{name}_loader") for name in ("train", "valid", "test")}
    try:
        _draw_all(predictions, history, everything, plots, keys, predicts, figures, bus, loaders, record)
    finally:
        figure.configure(before)
    if plots:
        logger.info(f"plots: {', '.join(plots)}")
    return None


def _draw_all(predictions, history, everything, plots, keys, predicts, figures, bus, loaders, record):
    for name, plot in (plots or {}).items():
        logger.debug(f"drawing {name}")
        definition = keys.get(name) or {}
        extra = plot_inputs(plot, definition.get("inputs"), predictions, history, everything)
        for key, value in bus.items():
            if _accepts(plot, key):
                extra[key] = value
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


def first_set(sets, default="train"):
    names = [name for name in (sets or []) if name]
    return names[0] if names else default


def set_frame(loaders, prep, set_name="train"):
    """The columns of one set in their original units: the features rescaled, the single column targets inverted
    and the extra columns as they were read. None when the set is not a table (images, a stream)."""
    import pandas

    loader = (loaders or {}).get(set_name)
    frame = getattr(getattr(loader, "dataset", None), "frame", None)
    data = getattr(frame, "data", None)
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
    extra = getattr(frame, "extra", None)
    for column in (list(extra.columns) if extra is not None else []):
        if column not in table.columns:
            table[column] = extra[column].to_numpy()
    return table if len(table.columns) else None


def columns_of(table, patterns=None, skip=()):
    """The numeric columns a pattern list names, in the table's order; every one of them without patterns."""
    import fnmatch

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


def _logged(values, column, log):
    if column not in set(log or ()):
        return values, column
    floor = numpy.nanpercentile(numpy.abs(values[values != 0]), 1) if numpy.any(values != 0) else 1e-12
    return numpy.log10(numpy.clip(values, max(float(floor), 1e-300), None)), f"log10({column})"


@lego("/plot/kalfa/target_vs_features", partial=True, alias="target_vs_features", refs={"target": "field"},
            description="One panel per feature: the target against it as a hexbin density with the median profile "
                        "over equal count bins; it reads the set the definition names (train without one) and "
                        "draws in the original units")
def target_vs_features(predictions, history, models, record, loaders=None, prep=None, sets=None, target=None,
                       columns=None, log=None, gridsize=60, bins=60, limit=24, per_row=4, name=None):
    set_name = first_set(sets, "train")
    table = set_frame(loaders, prep, set_name)
    if table is None:
        return None
    field = target or next(iter(prep.targets), None)
    if field is None or field not in table.columns:
        return None
    picked = columns_of(table, columns, skip=(field,) + tuple(prep.targets))[:int(limit)]
    if not picked:
        return None
    width = max(1, min(int(per_row or 4), len(picked)))
    rows = -(-len(picked) // width)
    drawing, axes = figure.grid(rows, width, width=6.4, height=4.2)
    panels = [axis for row in axes for axis in row]
    truth = table[field].to_numpy(dtype="float64")
    for axis, column in zip(panels, picked):
        values, shown = _logged(table[column].to_numpy(dtype="float64"), column, log)
        x, y = figure.finite(values, truth)
        if not len(x):
            axis.axis("off")
            continue
        figure.density(drawing, axis, x, y, gridsize)
        centers, profile = figure.profile(x, y, bins)
        if len(centers):
            axis.plot(centers, profile, color=figure.CATEGORICAL[1], linewidth=2.4, label="median profile")
            axis.legend(loc="lower left")
        figure.label(axis, f"{field} vs {shown}", shown, field)
    for axis in panels[len(picked):]:
        axis.axis("off")
    figure.title(drawing, f"{field} against every feature ({set_name} set, {len(table):,} rows)")
    drawing.tight_layout(rect=(0, 0, 1, 0.97))
    figure.save(drawing, record, name or "target_vs_features")
    return None


@lego("/plot/kalfa/correlation_heatmap", partial=True, alias="correlation_heatmap",
            description="The rank correlation of every column of a set against every other, features and targets "
                        "together; it reads the set the definition names (train without one)")
def correlation_heatmap(predictions, history, models, record, loaders=None, prep=None, sets=None,
                        method="spearman", columns=None, sample=80000, annotate=False, name=None):
    set_name = first_set(sets, "train")
    table = set_frame(loaders, prep, set_name)
    if table is None:
        return None
    picked = columns_of(table, columns)
    if len(picked) < 2:
        return None
    data = table[picked]
    if sample and len(data) > int(sample):
        data = data.sample(int(sample), random_state=0)
    matrix = data.corr(method=method).to_numpy()
    side = 0.34 * len(picked) + 3.4
    drawing, axes = figure.sized(side, side * 0.92)
    axis = axes[0][0]
    drawn = axis.imshow(matrix, cmap=figure.diverging(), vmin=-1, vmax=1)
    axis.set_xticks(range(len(picked)), picked, rotation=90, fontsize=8)
    axis.set_yticks(range(len(picked)), picked, fontsize=8)
    axis.grid(visible=False)
    if annotate and len(picked) <= 20:
        for row in range(len(picked)):
            for column in range(len(picked)):
                axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", fontsize=7,
                          color=figure.INK if abs(matrix[row, column]) < 0.6 else "#ffffff")
    figure.colorbar(drawing, drawn, axis, f"{method} rho", fraction=0.032)
    figure.label(axis, "Column correlation", None, None,
                 note=f"{len(data):,} rows of the {set_name} set, {len(picked)} columns")
    figure.save(drawing, record, name or "correlation_heatmap")
    return None


def _points(count):
    return f"{int(count)} point" + ("" if int(count) == 1 else "s")


def pick_pair(predictions, output=None, target=None):
    """The (prediction column, target column) a definition names: by output wire, by target field, or the first."""
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


@lego("/plot/kalfa/residuals", partial=True, alias="residuals", refs={"target": "field"},
            description="Three panels of one prediction's residual: the distribution with its bias and sigma, the "
                        "residual against the truth as a density, and the mean and median error over equal count "
                        "bins of the target range")
def residuals(predictions, history, models, record, output=None, target=None, bins=20, gridsize=60, name=None):
    pred, field = pick_pair(predictions, output, target)
    if pred is None:
        return None
    truth, guess = figure.finite(predictions[field].to_numpy(), predictions[pred].to_numpy())
    if len(truth) < 2:
        return None
    error = guess - truth
    drawing, axes = figure.grid(1, 3, width=5.0, height=4.0)
    left, middle, right = axes[0]

    left.hist(error, bins=140, color=figure.CATEGORICAL[0], edgecolor="none")
    left.axvline(0.0, color=figure.INK_MUTED, linewidth=1)
    figure.label(left, "Residual distribution", "predicted - true", "points",
                 note=f"bias {error.mean():+.4f}, sigma {error.std():.4f}")

    figure.density(drawing, middle, truth, error, gridsize)
    middle.axhline(0.0, color=figure.CATEGORICAL[1], linewidth=2)
    figure.label(middle, "Residual vs truth", f"true {field}", "residual")

    centers, median = figure.profile(truth, numpy.abs(error), bins)
    _, mean = figure.profile(truth, numpy.abs(error), bins, statistic="mean")
    if len(centers):
        right.plot(centers, median, marker="o", color=figure.CATEGORICAL[0], label="median |error|")
        right.plot(centers, mean, marker="s", color=figure.CATEGORICAL[1], label="mean |error|")
        right.legend(loc="upper left")
    figure.label(right, "Error across the target range", f"true {field} (equal count bins)", "|residual|")

    figure.title(drawing, f"{field} residuals ({len(truth):,} test points)")
    drawing.tight_layout(rect=(0, 0, 1, 0.94))
    figure.save(drawing, record, name or "residuals")
    return None


@lego("/plot/kalfa/error_map", partial=True, alias="error_map",
            refs={"x": "column", "y": "column", "target": "field"},
            description="The error of one prediction over a 2d grid of two columns: with statistic residual blue "
                        "is a prediction below the truth and red above it, with abs the mean absolute error; bins "
                        "holding fewer than min_count points stay empty")
def error_map(predictions, history, models, record, loaders=None, prep=None, sets=None, x=None, y=None,
              output=None, target=None, statistic="residual", bins=55, min_count=15, name=None):
    pred, field = pick_pair(predictions, output, target)
    table = set_frame(loaders, prep, first_set(sets, "test"))
    if pred is None or table is None or x is None or y is None:
        return None
    if x not in table.columns or y not in table.columns or "row" not in predictions.columns:
        return None
    picked = table.reindex(predictions["row"].to_numpy())
    error = predictions[pred].to_numpy(dtype="float64") - predictions[field].to_numpy(dtype="float64")
    if statistic == "abs":
        error = numpy.abs(error)
    across, along, values = figure.finite(picked[x].to_numpy(), picked[y].to_numpy(), error)
    if len(across) < int(min_count):
        return None
    x_edges, y_edges, mean = figure.binned(across, along, values, bins, min_count)
    drawing, axis = figure.single(width=7.0, height=5.0)
    if statistic == "abs":
        drawn = axis.pcolormesh(x_edges, y_edges, mean.T, cmap=figure.sequential(), shading="auto")
        note = f"mean absolute error; bins with at least {_points(min_count)}"
        text = "mean |error|"
    else:
        limit = figure.symmetric(mean)
        drawn = axis.pcolormesh(x_edges, y_edges, mean.T, cmap=figure.diverging(), vmin=-limit, vmax=limit,
                                shading="auto")
        note = (f"blue: prediction below truth, red: prediction above truth; bins with at least "
                f"{_points(min_count)}")
        text = "mean residual (pred - true)"
    figure.colorbar(drawing, drawn, axis, text)
    axis.grid(visible=False)
    figure.label(axis, f"Where the model is off, over ({x}, {y})", x, y, note=note)
    figure.save(drawing, record, name or "error_map")
    return None


def _feature_batch(loader, model, prep, sample):
    """One matrix of the model's first input wire and the target tensor of the pass, up to sample rows."""
    import torch

    from .runtime import model_inputs, named_outputs

    wires = model_inputs(model)
    device = next(iter(model.parameters()), torch.zeros(1)).device
    features, targets, taken = [], [], 0
    names = list(prep.targets) if prep is not None else []
    for batch in loader:
        if wires[0] not in batch or not names or names[0] not in batch:
            return None, None, None
        features.append(batch[wires[0]])
        targets.append(batch[names[0]])
        taken += len(features[-1])
        if taken >= int(sample):
            break
    if not features:
        return None, None, None
    matrix = torch.cat(features)[:int(sample)].to(device)
    truth = torch.cat(targets)[:int(sample)].to(device).reshape(len(matrix), -1)
    if matrix.ndim != 2:
        return None, None, None
    return matrix, truth, named_outputs


def _scored(model, matrix, truth, wire, named_outputs):
    import torch

    with torch.no_grad():
        outputs = named_outputs(model, model(matrix))
    guess = outputs[wire] if wire in outputs else next(iter(outputs.values()))
    guess = guess.reshape(len(matrix), -1)[:, :truth.shape[1]]
    spread = float(((truth - truth.mean(dim=0)) ** 2).sum())
    if spread <= 0:
        return float("nan")
    return 1.0 - float(((guess - truth) ** 2).sum()) / spread


@lego("/plot/kalfa/permutation_importance", partial=True, alias="permutation_importance",
            description="The drop in R2 when one feature column is shuffled, the largest first; the model runs "
                        "again for every feature and every repeat, so sample bounds the cost")
def permutation_importance(predictions, history, models, record, loaders=None, prep=None, predicts=None, sets=None,
                           repeats=3, sample=20000, top=25, output=None, groups=None, seed=0, name=None):
    import torch

    from .runtime import resolve_model

    loader = (loaders or {}).get(first_set(sets, "test"))
    if loader is None or prep is None or predicts is None:
        return None
    model = resolve_model(predicts, models)
    model.eval()
    matrix, truth, tools = _feature_batch(loader, model, prep, sample)
    if matrix is None or matrix.shape[1] != len(prep.features):
        return None
    wire = output or ""
    base = _scored(model, matrix, truth, wire, tools)
    if not numpy.isfinite(base):
        return None
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    means, deviations = [], []
    for column in range(matrix.shape[1]):
        scores = []
        for _ in range(int(repeats)):
            shuffled = matrix.clone()
            order = torch.randperm(len(matrix), generator=generator).to(matrix.device)
            shuffled[:, column] = matrix[order, column]
            scores.append(base - _scored(model, shuffled, truth, wire, tools))
        means.append(float(numpy.mean(scores)))
        deviations.append(float(numpy.std(scores)))
    order = numpy.argsort(means)[::-1][:int(top)][::-1]
    names = [prep.features[position] for position in order]
    values = [means[position] for position in order]
    spread = [deviations[position] for position in order]
    drawing, axes = figure.sized(figure.width_of(9.5), 0.34 * len(names) + 2.0)
    axis = axes[0][0]
    bars(axis, names, values, groups, spread)
    figure.label(axis, "Permutation importance", "drop in R2 when the feature is shuffled", None,
                 note=f"R2 = {base:.4f} on {len(matrix):,} points, {int(repeats)} repeats")
    figure.save(drawing, record, name or "permutation_importance")
    return None


def bars(axis, names, values, groups=None, spread=None):
    """Horizontal bars coloured by the group a name belongs to; returns the group order for the legend."""
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


@lego("/plot/kalfa/feature_distributions", partial=True, alias="feature_distributions",
            description="A histogram per feature column of a set, in the original units; log names the columns to "
                        "draw on a log10 axis")
def feature_distributions(predictions, history, models, record, loaders=None, prep=None, sets=None, columns=None,
                          log=None, bins=80, limit=24, per_row=4, name=None):
    set_name = first_set(sets, "train")
    table = set_frame(loaders, prep, set_name)
    if table is None:
        return None
    picked = columns_of(table, columns)[:int(limit)]
    if not picked:
        return None
    width = max(1, min(int(per_row or 4), len(picked)))
    rows = -(-len(picked) // width)
    drawing, axes = figure.grid(rows, width, width=3.4, height=2.6)
    panels = [axis for row in axes for axis in row]
    for axis, column in zip(panels, picked):
        values, shown = _logged(table[column].to_numpy(dtype="float64"), column, log)
        values = figure.finite(values)[0]
        if not len(values):
            axis.axis("off")
            continue
        axis.hist(values, bins=int(bins), color=figure.CATEGORICAL[0], edgecolor="none")
        axis.tick_params(labelsize=8)
        figure.label(axis, shown, None, None)
    for axis in panels[len(picked):]:
        axis.axis("off")
    figure.title(drawing, f"Column distributions ({set_name} set, {len(table):,} rows)")
    drawing.tight_layout(rect=(0, 0, 1, 0.97))
    figure.save(drawing, record, name or "feature_distributions")
    return None


@lego("/plot/kalfa/target_correlation", partial=True, alias="target_correlation", refs={"target": "field"},
            description="The rank correlation of every column with the target, the strongest first; groups maps a "
                        "column to a group name and colours the bars by it")
def target_correlation(predictions, history, models, record, loaders=None, prep=None, sets=None, target=None,
                       method="spearman", columns=None, top=25, groups=None, name=None):
    set_name = first_set(sets, "train")
    table = set_frame(loaders, prep, set_name)
    if table is None:
        return None
    field = target or next(iter(prep.targets), None)
    if field is None or field not in table.columns:
        return None
    picked = columns_of(table, columns, skip=(field,) + tuple(prep.targets))
    if not picked:
        return None
    truth = table[field]
    found = [(column, float(table[column].corr(truth, method=method))) for column in picked]
    found = [(column, value) for column, value in found if numpy.isfinite(value)]
    found.sort(key=lambda item: abs(item[1]), reverse=True)
    found = found[:int(top)][::-1]
    if not found:
        return None
    names = [column for column, _ in found]
    drawing, axes = figure.sized(figure.width_of(9.5), 0.34 * len(names) + 2.0)
    axis = axes[0][0]
    bars(axis, names, [value for _, value in found], groups)
    figure.label(axis, f"Rank correlation with {field}", f"{method} rho", None,
                 note=f"{len(table):,} rows of the {set_name} set, the {len(names)} strongest")
    figure.save(drawing, record, name or "target_correlation")
    return None


def _seaborn(what):
    try:
        import seaborn
    except ImportError:
        logger.warning(f"{what}: seaborn is not installed, so the plot is skipped; pip install seaborn for it")
        return None
    return seaborn


def _sampled(table, sample, seed=0):
    if sample and len(table) > int(sample):
        return table.sample(int(sample), random_state=int(seed))
    return table


@lego("/plot/seaborn/pairplot", partial=True, alias="pairplot",
            description="seaborn's pairwise grid of a few columns of a set, hue colouring the points by a column; "
                        "skipped with a warning when seaborn is not installed")
def pairplot(predictions, history, models, record, loaders=None, prep=None, sets=None, columns=None, hue=None,
             sample=5000, kind="scatter", diagonal="hist", height=2.2, name=None):
    seaborn = _seaborn("pairplot")
    set_name = first_set(sets, "train")
    table = set_frame(loaders, prep, set_name)
    if seaborn is None or table is None:
        return None
    picked = columns_of(table, columns)[:8]
    if len(picked) < 2:
        return None
    wanted = picked + ([hue] if hue and hue in table.columns and hue not in picked else [])
    data = _sampled(table[wanted], sample)
    grid = seaborn.pairplot(data, vars=picked, hue=hue if hue in data.columns else None, kind=kind,
                            diag_kind=diagonal, height=float(height),
                            palette=figure.CATEGORICAL if hue in data.columns else None,
                            plot_kws={"color": figure.CATEGORICAL[0], "edgecolor": "none", "s": 12}
                            if hue not in data.columns else None)
    figure.title(grid.figure, f"Pairwise columns ({set_name} set, {len(data):,} rows)")
    figure.save(grid.figure, record, name or "pairplot")
    return None


@lego("/plot/seaborn/violin", partial=True, alias="violin", refs={"value": "column", "group": "column"},
            description="seaborn's violin of one column of a set, split by a grouping column when one is named; "
                        "skipped with a warning when seaborn is not installed")
def violin(predictions, history, models, record, loaders=None, prep=None, sets=None, value=None, group=None,
           sample=20000, name=None):
    seaborn = _seaborn("violin")
    set_name = first_set(sets, "train")
    table = set_frame(loaders, prep, set_name)
    if seaborn is None or table is None or value is None or value not in table.columns:
        return None
    data = _sampled(table, sample)
    drawing, axis = figure.single(width=7.0, height=4.6)
    seaborn.violinplot(data=data, x=group if group in data.columns else None, y=value, ax=axis,
                       color=figure.CATEGORICAL[0], palette=figure.CATEGORICAL if group in data.columns else None,
                       hue=group if group in data.columns else None, legend=False)
    figure.label(axis, f"{value} by {group}" if group in data.columns else value, group, value,
                 note=f"{len(data):,} rows of the {set_name} set")
    figure.save(drawing, record, name or "violin")
    return None


@lego("/plot/seaborn/kde", partial=True, alias="kde", refs={"x": "column", "y": "column", "hue": "column"},
            description="seaborn's kernel density of one column of a set, or of two as contours; skipped with a "
                        "warning when seaborn is not installed")
def kde(predictions, history, models, record, loaders=None, prep=None, sets=None, x=None, y=None, hue=None,
        sample=20000, fill=True, name=None):
    seaborn = _seaborn("kde")
    set_name = first_set(sets, "train")
    table = set_frame(loaders, prep, set_name)
    if seaborn is None or table is None or x is None or x not in table.columns:
        return None
    data = _sampled(table, sample)
    drawing, axis = figure.single(width=6.4, height=4.6)
    seaborn.kdeplot(data=data, x=x, y=y if y in data.columns else None, hue=hue if hue in data.columns else None,
                    fill=bool(fill), ax=axis, color=figure.CATEGORICAL[0],
                    palette=figure.CATEGORICAL if hue in data.columns else None)
    figure.label(axis, f"{x} and {y}" if y in data.columns else f"{x} density", x,
                 y if y in data.columns else "density", note=f"{len(data):,} rows of the {set_name} set")
    figure.save(drawing, record, name or "kde")
    return None
