import numpy

from kalfa.registration import lego
from kalfa.std.common.figure import Figure
from kalfa.std.pre.base import TableFrame


def count(value):
    return "?" if value is None else f"{value:,}"


def stage_lines(report):
    lines = []
    stages = report.get("stages") or []
    if stages:
        first = stages[0]
        lines.append(("source", f"{count(first['rows'])} rows x {first['columns']} columns"))
    for entry in stages[1:]:
        change = []
        if entry.get("added"):
            change.append("+" + ", ".join(entry["added"]))
        if entry.get("removed"):
            change.append("-" + ", ".join(entry["removed"]))
        lines.append(("transform", f"{entry['stage']}: {count(entry['rows'])} rows x {entry['columns']} columns"
                                   + (f"  ({'; '.join(change)})" if change else "")))
    split = report.get("split") or {}
    if split:
        lines.append(("split", " | ".join(f"{name} {count(rows)}" for name, rows in split.items())))
    after = report.get("after_set_transforms") or {}
    if after and after != split:
        lines.append(("transform", "after the set transforms: "
                                   + " | ".join(f"{name} {count(rows)}" for name, rows in after.items())))
    if report.get("frames"):
        lines.append(("frames", ", ".join(report["frames"]) + "  fitted on train"))
    fit = report.get("fit") or {}
    fitted = ", ".join(f"{name} {columns}" for name, columns in (fit.get("preprocessors") or {}).items())
    lines.append(("fit", f"{fitted or 'no preprocessors'} -> {fit.get('features', 0)} features, "
                         f"{len(fit.get('targets') or [])} targets"
                         + (f", extras {', '.join(fit['extras'])}" if fit.get("extras") else "")))
    sets = report.get("sets") or {}
    if sets:
        lines.append(("feed", " | ".join(f"{name} {count(entry['rows'])} rows" for name, entry in sets.items())))
    loaders = report.get("loaders") or {}
    if loaders:
        lines.append(("loaders", " | ".join(f"{name} {count(entry['batches'])} x {entry['size']}"
                                            for name, entry in loaders.items())))
    return lines


def histogram_columns(prep, train_df, train_frame, columns):
    if prep is None or train_df is None or not isinstance(train_frame, TableFrame):
        return []
    items = [item for item in prep.fields if item.columns and item.columns[0] in train_frame.data.columns
             and item.name in train_df.columns]
    if columns:
        items = [item for item in items if item.name in columns]
    else:
        items = sorted(items, key=lambda item: -len(item.chain))[:3]
    chosen = []
    for item in items:
        before = train_df[item.name].to_numpy()
        after = train_frame.data[item.columns[0]].to_numpy()
        if before.dtype.kind in "fiu" and after.dtype.kind in "fiub":
            chosen.append((item, before.astype("float64"), after.astype("float64")))
    return chosen


def draw_report(figures, report, histograms, name):
    lines = stage_lines(report)
    columns = max(2 * len(histograms), 1)
    rows = 2 if histograms else 1
    pyplot = figures.pyplot()
    drawing = pyplot.figure(figsize=(max(9.0, 3.0 * columns), 0.42 * len(lines) + (3.4 if histograms else 1.0)))
    grid = drawing.add_gridspec(rows, columns, height_ratios=[0.42 * len(lines), 2.6] if histograms else [1.0])
    top = drawing.add_subplot(grid[0, :])
    palette = figures.categorical
    colors = {"source": figures.ink_muted, "transform": palette[2], "split": palette[0], "frames": palette[6],
              "fit": palette[3], "feed": palette[4], "loaders": palette[1]}
    for position, (kind, text) in enumerate(lines):
        y = 1.0 - (position + 0.5) / len(lines)
        top.text(0.0, y, kind, ha="left", va="center", fontsize=8.5, fontweight="bold", color=colors[kind],
                 transform=top.transAxes)
        top.text(0.11, y, text, ha="left", va="center", fontsize=8.5, color=figures.ink, family="monospace",
                 transform=top.transAxes)
    top.axis("off")
    for position, (item, before, after) in enumerate(histograms):
        left = drawing.add_subplot(grid[1, 2 * position])
        right = drawing.add_subplot(grid[1, 2 * position + 1])
        left.hist(before[numpy.isfinite(before)], bins=40, color=figures.ink_muted)
        right.hist(after[numpy.isfinite(after)], bins=40, color=palette[0])
        figures.label(left, title=f"{item.name} before", ylabel="rows" if position == 0 else None)
        figures.label(right, title=f"{item.columns[0]} after {' > '.join(item.chain)}")
    figures.title(drawing, name)
    return drawing


@lego("/plot/kalfa/data_pipeline", partial=True, alias="data_pipeline", needs=["data_report"],
      description="The data block as one picture: every stage with its rows and columns, the split, the fitted "
                  "frame transforms and preprocessors, the features and targets, the loaders; under the fit, "
                  "before and after histograms of the columns with the longest chains (columns names others) "
                  "when the source is a table")
def data_pipeline(predictions, history, models, record, data_report=None, train_df=None, train_frame=None, prep=None,
                  columns=None, name=None, figures=None):
    figures = figures or Figure()
    if not data_report:
        return None
    histograms = histogram_columns(prep, train_df, train_frame, columns)
    stem = name or "data_pipeline"
    figures.save(draw_report(figures, data_report, histograms, stem), record, stem)
    return None
