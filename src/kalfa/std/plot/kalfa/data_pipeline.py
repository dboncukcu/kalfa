import textwrap

import numpy

from kalfa.std.common.figure import Figure
from kalfa.std.pre.base import TableFrame


def count(value):
    return "?" if value is None else f"{value:,}"


def named(names, limit=10):
    names = list(names)
    if len(names) <= limit:
        return ", ".join(names)
    return ", ".join(names[:limit]) + f" and {len(names) - limit} more"


def stage_lines(report):
    lines = []
    stages = report.get("stages") or []
    if stages:
        first = stages[0]
        lines.append(("source", f"{count(first['rows'])} rows x {first['columns']} columns"))
    for entry in stages[1:]:
        change = []
        if entry.get("added"):
            change.append("+" + named(entry["added"]))
        if entry.get("removed"):
            change.append("-" + named(entry["removed"]))
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


def wrapped(lines, width=96):
    rows = []
    for kind, text in lines:
        parts = textwrap.wrap(text, width=width, subsequent_indent="    ") or [""]
        rows.append((kind, parts[0]))
        rows.extend(("", part) for part in parts[1:])
    return rows


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
    rows = wrapped(stage_lines(report))
    pyplot = figures.pyplot()
    text_height = 0.22 * len(rows) + 0.55
    panel = 2.4
    drawing = pyplot.figure(figsize=(10.0, text_height + panel * len(histograms) + 0.5))
    grid = drawing.add_gridspec(1 + len(histograms), 2, height_ratios=[text_height, *[panel] * len(histograms)],
                                hspace=0.6, wspace=0.28, left=0.07, right=0.98, top=0.93, bottom=0.08)
    top = drawing.add_subplot(grid[0, :])
    palette = figures.categorical
    colors = {"source": figures.ink_muted, "transform": palette[2], "split": palette[0], "frames": palette[6],
              "fit": palette[3], "feed": palette[4], "loaders": palette[1], "": figures.ink}
    for position, (kind, text) in enumerate(rows):
        y = 1.0 - (position + 0.5) / len(rows)
        if kind:
            top.text(0.0, y, kind, ha="left", va="center", fontsize=8.5, fontweight="bold", color=colors[kind],
                     transform=top.transAxes)
        top.text(0.09, y, text, ha="left", va="center", fontsize=8, color=figures.ink, family="monospace",
                 transform=top.transAxes)
    top.axis("off")
    for position, (item, before, after) in enumerate(histograms):
        left = drawing.add_subplot(grid[1 + position, 0])
        right = drawing.add_subplot(grid[1 + position, 1])
        left.hist(before[numpy.isfinite(before)], bins=40, color=figures.ink_muted)
        right.hist(after[numpy.isfinite(after)], bins=40, color=palette[0])
        figures.label(left, title=f"{item.name} before", ylabel="rows", note="original units")
        figures.label(right, title=f"{item.columns[0]} after", note=" > ".join(item.chain))
    figures.title(drawing, name)
    return drawing


def data_pipeline(predictions, history, models, record, data_report=None, train_df=None, train_frame=None, prep=None,
                  columns=None, name=None, figures=None):
    figures = figures or Figure()
    if not data_report:
        return None
    histograms = histogram_columns(prep, train_df, train_frame, columns)
    stem = name or "data_pipeline"
    figures.save(draw_report(figures, data_report, histograms, stem), record, stem, tight=False)
    return None
