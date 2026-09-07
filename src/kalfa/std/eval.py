"""Evaluation: per set metrics, the test predictions and generation after training."""

from pathlib import Path

import numpy
import torch

from ..registration import lego
from .feed import sized

from .runtime import (Context, active_entries, collect_results, named_outputs, call_model, observe_all, resolve_model,
                      set_modes, to_device, tracker_for, turn_generator)


@lego("/lego/kalfa/evaluate", returns="metrics", bus=["device", "prep", "record"],
            description="Losses (model scale) and metrics (original scale, through prep) of one set under no_grad; "
                        "an empty set gives an empty mapping; record reaches metrics that write files")
def evaluate(models, emas, composites, counters, effects, loader, set, losses, metrics, losses_keys, metrics_keys,
             predicts, device=None, prep=None, record=None):
    if loader is None or sized(loader.dataset) == 0:
        return {}
    turn = int(counters.get("turn", 0))
    trackers = [tracker_for(name, entry, keys) for name, entry, keys in active_entries(losses, losses_keys, set, turn)]
    trackers += [tracker_for(name, entry, keys, rescale=True)
                 for name, entry, keys in active_entries(metrics, metrics_keys, set, turn)]
    if not trackers:
        return {}
    targets = list(getattr(loader.dataset, "targets", []))
    set_modes(models, train=False, composites=composites)
    rng = turn_generator(device)
    seen = False
    with torch.no_grad():
        for batch in loader:
            seen = True
            context = Context(to_device(batch, device), models, composites, emas, predicts, targets,
                              step=counters.get("global_step", 0), epoch=turn, rng=rng, prep=prep, set_name=set,
                              losses=losses, losses_keys=losses_keys, record=record)
            observe_all(trackers, context)
    if not seen:
        return {}
    return collect_results(trackers)


def prediction_table(model, loader, prep, dataset, device=None):
    """The predictions DataFrame: row id, targets inverted, pred_<wire> inverted or decoded, raw_<wire>."""
    import pandas

    model.eval()
    raw = {}
    observed = {name: [] for name in dataset.targets}
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            outputs = named_outputs(model, call_model(model, batch))
            for wire, value in outputs.items():
                raw.setdefault(wire, []).append(value.detach().cpu().numpy().reshape(len(value), -1))
            for name in dataset.targets:
                value = batch[name]
                observed[name].append(value.detach().cpu().numpy().reshape(len(value), -1))
    set_name = dataset.frame.set
    columns = {"row": numpy.asarray(dataset.rows())}
    widths = {}
    for name in dataset.targets:
        values = numpy.concatenate(observed[name]) if observed[name] else numpy.zeros((0, 1))
        widths[name] = values.shape[1]
        if values.shape[1] == 1:
            columns[name] = prep.inverse(name, values[:, 0], set_name)
        else:
            for position in range(values.shape[1]):
                columns[f"{name}_{position}"] = prep.inverse(name, values[:, position], set_name)
    total = sum(widths.values())
    single = dataset.targets[0] if len(dataset.targets) == 1 else None
    for wire, pieces in raw.items():
        matrix = numpy.concatenate(pieces) if pieces else numpy.zeros((0, 1))
        width = matrix.shape[1]
        if width == 1:
            columns[f"raw_{wire}"] = matrix[:, 0]
        else:
            for position in range(width):
                columns[f"raw_{wire}_{position}"] = matrix[:, position]
        decoder = prep.decoder(single) if single is not None else None
        if decoder is not None:
            columns[f"pred_{wire}"] = prep.decode(single, matrix, set_name)
        elif width == total and total:
            offset = 0
            for name in dataset.targets:
                span = widths[name]
                block = matrix[:, offset:offset + span]
                offset += span
                label = f"pred_{wire}" if single is not None else f"pred_{wire}_{name}"
                if span == 1:
                    columns[label] = prep.inverse(name, block[:, 0], set_name)
                else:
                    for position in range(span):
                        columns[f"{label}_{position}"] = prep.inverse(name, block[:, position], set_name)
    return pandas.DataFrame(columns)


@lego("/lego/kalfa/predict", returns="predictions", bus=["record", "device"],
            description="Predict the test set with the report model, invert the target chain, "
                        "write predictions.parquet")
def predict(models, composites, loader, prep, predicts, set, record=None, device=None):
    import pandas

    if loader is None or sized(loader.dataset) == 0 or predicts is None:
        return pandas.DataFrame()
    model = resolve_model(predicts, models, composites)
    table = prediction_table(model, loader, prep, loader.dataset, device)
    if len(table) == 0:
        return pandas.DataFrame()
    if record is not None:
        target = Path(record)
        target.mkdir(parents=True, exist_ok=True)
        table.to_parquet(target / "predictions.parquet", index=False)
    return table


@lego("/lego/kalfa/generate", returns=None, bus=["record"],
            description="Run the generate lego with the report models; nothing without a generate section")
def generate(models, composites, prep, generate, record=None):
    if generate is None:
        return None
    samples = generate(models={**dict(composites or {}), **dict(models)}, prep=prep, rng=turn_generator())
    if record is not None and samples is not None:
        write_samples(samples, Path(record) / "samples")
    return None


def write_samples(samples, target):
    """Samples under samples/: tensors as samples.pt (images also as grid.png), text as samples.txt."""
    write_sample_files(samples, target, "samples", "grid")


def write_turn_samples(samples, target, turn):
    """The samples of one turn under samples/: turn_<n>.pt and turn_<n>.png, or turn_<n>.txt for text."""
    stem = f"turn_{int(turn):04d}"
    write_sample_files(samples, target, stem, stem)


def write_sample_files(samples, target, stem, image_stem):
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    if isinstance(samples, str):
        (target / f"{stem}.txt").write_text(samples)
        return
    torch.save(samples, target / f"{stem}.pt")
    if isinstance(samples, torch.Tensor) and samples.ndim == 4:
        write_grid(samples, target / f"{image_stem}.png")


def write_grid(images, path):
    """An image grid of at most eight columns, saved as png."""
    from .plot import _figure, _image_grid

    count = len(images)
    columns = min(8, count)
    rows = (count + columns - 1) // columns
    pyplot = _figure()
    figure, axes = pyplot.subplots(rows, columns, figsize=(1.6 * columns, 1.6 * rows), squeeze=False)
    for position in range(rows * columns):
        axis = axes[position // columns][position % columns]
        if position < count:
            _image_grid(axis, images[position])
        else:
            axis.axis("off")
    figure.savefig(path, bbox_inches="tight")
    pyplot.close(figure)
