import re

import numpy

from kalfa.records import reader, stats
from kalfa.std.plots.common import color_of, directory_of, dress, label_text, quiet
from kalfa.std.plots.pairs import chosen_pair, pred_column, prediction_pairs, read_predictions


HORIZON = re.compile(r"(.+)_(\d+)")


def numeric(frame, column):
    return frame[column].dtype.kind in "fiu"


def r_squared(truth, guess):
    spread = float(numpy.sum((truth - truth.mean()) ** 2))
    if spread <= 0:
        return None
    return 1.0 - float(numpy.sum((guess - truth) ** 2)) / spread


def pred_vs_true(record, figures, pairs=None):
    directory = directory_of(record)
    frame = read_predictions(directory)
    if frame is None:
        return None
    chosen = [(pred, target) for pred, target in prediction_pairs(frame, quiet(reader.read_plan, directory), pairs)
              if numeric(frame, pred) and numeric(frame, target)]
    if not chosen:
        return None
    columns = min(3, len(chosen))
    rows = -(-len(chosen) // columns)
    figure = figures.new(4.8 * columns, 4.4 * rows)
    panels = [axis for row in figure.subplots(rows, columns, squeeze=False) for axis in row]
    for axis, (pred, target) in zip(panels, chosen):
        truth = frame[target].to_numpy(dtype="float64", na_value=numpy.nan)
        guess = frame[pred].to_numpy(dtype="float64", na_value=numpy.nan)
        keep = numpy.isfinite(truth) & numpy.isfinite(guess)
        truth, guess = truth[keep], guess[keep]
        if not len(truth):
            axis.axis("off")
            continue
        axis.scatter(truth, guess, s=12, alpha=0.5, color=color_of(figures, 0), edgecolors="none", rasterized=True)
        low = float(min(truth.min(), guess.min()))
        high = float(max(truth.max(), guess.max()))
        axis.plot([low, high], [low, high], color=color_of(figures, 1), linewidth=1.5, label="y = x")
        axis.legend(loc="upper left", frameon=False)
        score = r_squared(truth, guess)
        note = f"R² {score:.3f}, " if score is not None else ""
        axis.set_title(f"{target} ({note}n = {len(truth):,})", loc="left")
        axis.set_xlabel(f"true {target}")
        axis.set_ylabel(f"predicted ({pred[len('pred_'):]})")
        dress(axis)
    for axis in panels[len(chosen):]:
        axis.axis("off")
    figures.save(figure, "pred_vs_true")
    return None


def indexed(columns, name):
    found = {}
    for column in columns:
        match = HORIZON.fullmatch(column)
        if match is not None and match.group(1) == name:
            found[int(match.group(2))] = column
    return found


def horizon_pair(frame, plan, pred=None, target=None):
    columns = [str(column) for column in frame.columns]
    stems = []
    for column in columns:
        match = HORIZON.fullmatch(column)
        if match is not None and match.group(1) not in stems:
            stems.append(match.group(1))
    if pred is not None:
        outputs = [pred_column(pred)[len("pred_"):]]
    else:
        outputs = [stem[len("pred_"):] for stem in stems if stem.startswith("pred_")]
    known = [stem for stem in stems if not stem.startswith(("pred_", "raw_", "prob_"))]
    pairings = stats.plan_pairings(plan)
    for output in outputs:
        guesses = indexed(columns, f"pred_{output}")
        names = [str(target)] if target is not None else [pairings.get(output), *stats.plan_targets(plan), *known]
        for name in names:
            truths = indexed(columns, name) if name else {}
            steps = sorted(step for step in guesses if step in truths)
            if steps:
                return output, name, [truths[step] for step in steps], [guesses[step] for step in steps]
    return None


def forecast_series(frame, plan, fitted, figures, pred, target):
    pair = chosen_pair(frame, plan, fitted, pred, target, prefer="regression")
    if pair is None or not numeric(frame, pair["pred"]) or not numeric(frame, pair["target"]):
        return None
    rows = frame["row"].to_numpy() if "row" in frame.columns else numpy.arange(len(frame))
    truth = frame[pair["target"]].to_numpy(dtype="float64", na_value=numpy.nan)
    guess = frame[pair["pred"]].to_numpy(dtype="float64", na_value=numpy.nan)
    figure = figures.new(9.0, 4.2)
    axis = figure.subplots()
    axis.plot(rows, truth, color=color_of(figures, 0), linewidth=1.4, label="true")
    axis.plot(rows, guess, color=color_of(figures, 1), linewidth=1.4, linestyle="--", label="predicted")
    axis.legend(loc="upper right", frameon=False)
    axis.set_title(f"{pair['target']} over rows (n = {len(frame):,})", loc="left")
    axis.set_xlabel("row")
    axis.set_ylabel(f"{pair['target']} and {pair['output']}")
    dress(axis)
    figures.save(figure, "forecast_samples")
    return None


def forecast_samples(record, figures, n=4, pred=None, target=None):
    from matplotlib.ticker import MaxNLocator

    directory = directory_of(record)
    frame = stats.read_predictions(directory)
    if frame is None or not len(frame):
        return None
    plan = quiet(reader.read_plan, directory)
    if "row" in frame.columns:
        frame = frame.sort_values("row", kind="stable")
    found = horizon_pair(frame, plan, pred, target)
    if found is None:
        return forecast_series(frame, plan, quiet(reader.fields_of, directory), figures, pred, target)
    output, name, truths, guesses = found
    picked = numpy.unique(numpy.linspace(0, len(frame) - 1, min(max(int(n), 1), len(frame))).round().astype(int))
    columns = 1 if len(picked) == 1 else 2
    rows = -(-len(picked) // columns)
    figure = figures.new(5.2 * columns, 2.4 * rows + 0.8)
    panels = [axis for line in figure.subplots(rows, columns, squeeze=False) for axis in line]
    steps = numpy.arange(1, len(truths) + 1)
    truth = frame[truths].to_numpy(dtype="float64", na_value=numpy.nan)
    guess = frame[guesses].to_numpy(dtype="float64", na_value=numpy.nan)
    labels = frame["row"].to_numpy() if "row" in frame.columns else frame.index.to_numpy()
    for index, (axis, position) in enumerate(zip(panels, picked)):
        axis.plot(steps, truth[position], color=color_of(figures, 0), linewidth=1.8, marker="o", markersize=3,
                  label="true")
        axis.plot(steps, guess[position], color=color_of(figures, 1), linewidth=1.8, linestyle="--",
                  label="predicted")
        axis.set_title(f"row {label_text(labels[position])}", loc="left")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        if index + columns >= len(picked):
            axis.set_xlabel("steps ahead")
        dress(axis)
    for axis in panels[len(picked):]:
        axis.axis("off")
    panels[0].legend(loc="upper left", frameon=False)
    figure.suptitle(f"Forecast windows ({name} against {output}, {len(steps)} steps, {len(picked)} of "
                    f"{len(frame):,} rows)", x=0.02, ha="left")
    figures.save(figure, "forecast_samples")
    return None
