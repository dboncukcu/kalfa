import base64
import difflib
import html
import io
import json
import math
import statistics
from datetime import datetime
from pathlib import Path

import numpy
import pandas

from . import __version__
from .collect import checkpoint_monitor, line_at
from .labels import classes_of, same_label
from .record import Record, failure_text, read_resolved, readable_state, settled_objective
from .std.common.history import History, is_number
from .std.plot.base import r2_of, true_column

REPORT = "report"
TASKS = ("regression", "classification")
EFFICIENCIES = (0.5, 0.8, 0.9, 0.95)
REGRESSION_BEST = {"rmse": abs, "mae": abs, "r2": lambda value: -value, "bias": abs, "median residual": abs,
                   "response": lambda value: abs(value - 1), "resolution": abs, "relative resolution": abs,
                   "within 10 %": lambda value: -value}
CSS = """
:root { --ink: #16181d; --muted: #6b7079; --line: #dfe2e7; --soft: #f4f6f9; --accent: #2a78d6; --low: #1b8a5a;
        --high: #c0392b; }
body { margin: 0; font: 14px/1.5 -apple-system, "Segoe UI", Roboto, sans-serif; color: var(--ink); background: #fff; }
main { max-width: 1180px; margin: 0 auto; padding: 32px 24px 64px; }
h1 { font-size: 26px; margin: 0 0 4px; }
h2 { font-size: 20px; margin: 44px 0 10px; padding-top: 12px; border-top: 1px solid var(--line); }
h3 { font-size: 16px; margin: 26px 0 8px; }
p, li { max-width: 900px; }
.muted { color: var(--muted); }
nav ol { columns: 2; padding-left: 18px; }
nav a, a { color: var(--accent); text-decoration: none; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0; }
.chip { font: 12px ui-monospace, Menlo, monospace; padding: 2px 9px; border-radius: 999px; background: var(--soft);
        border: 1px solid var(--line); }
.scroll { overflow-x: auto; margin: 8px 0 16px; }
table { border-collapse: collapse; font-size: 12.5px; }
th, td { padding: 4px 10px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }
th { background: var(--soft); font-weight: 600; }
td.num { text-align: right; font-family: ui-monospace, Menlo, monospace; }
td.low { color: var(--low); font-weight: 700; }
td.high { color: var(--high); font-weight: 700; }
td.both { color: var(--accent); font-weight: 700; }
figure { margin: 14px 0 22px; }
figure img { max-width: 100%; border: 1px solid var(--line); border-radius: 6px; }
figcaption { color: var(--muted); font-size: 12.5px; margin-top: 4px; max-width: 900px; }
pre { background: var(--soft); border: 1px solid var(--line); border-radius: 6px; padding: 10px 12px; overflow-x: auto;
      font-size: 12px; }
details { margin: 8px 0; }
.note { background: #fff8e6; border: 1px solid #f1d8a0; border-radius: 6px; padding: 8px 12px; }
dl.terms dt { font-weight: 600; margin-top: 8px; }
dl.terms dd { margin: 2px 0 0 0; color: var(--muted); max-width: 900px; }
@media print { h2 { break-before: page; } figure, table { break-inside: avoid; } nav { display: none; } }
"""


class Progress:
    def __init__(self, listen=None):
        self.listen = listen
        self.done = 0
        self.total = 0

    def plan(self, total):
        self.total = max(int(total), self.done)

    def begin(self, text):
        if self.listen is not None:
            self.listen(self.done, max(self.total, self.done + 1), text)
        self.done += 1


def finite(value):
    return is_number(value) and math.isfinite(value)


def cell(value):
    if isinstance(value, (numpy.floating, numpy.integer)):
        value = value.item()
    if isinstance(value, float):
        return f"{value:.6g}" if math.isfinite(value) else str(value)
    if isinstance(value, (list, tuple)):
        return ", ".join(cell(item) for item in value)
    return "" if value is None else str(value)


def number(value):
    return float(value) if value is not None and math.isfinite(float(value)) else None


def keyed(items, flag):
    found = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"{flag} takes TARGET=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        found[key.strip()] = value.strip()
    return found


def point_ids(text):
    if text is None:
        return None
    if text.strip() == "all":
        return "all"
    try:
        return sorted({int(part) for part in text.split(",") if part.strip()})
    except ValueError:
        raise ValueError(f"--points takes point ids like 3,7,12, or all; got {text!r}") from None


def readable_json(record, name):
    try:
        return record.read_json(name)
    except ValueError:
        return None


def folders(path):
    return sorted(child for child in Path(path).iterdir() if child.is_dir())


def reached_of(history, objective):
    monitor = objective.get("monitor")
    if monitor and len(history):
        try:
            value, turn = history.best(monitor, objective["mode"], objective["at"])
            return {"monitor": monitor, "value": value, "turn": turn}
        except ValueError:
            pass
    last = history[-1] if len(history) else {}
    return {"monitor": monitor, "value": None, "turn": last.get("turn")}


def entry_of(directory, label, values, objective):
    record = Record(directory)
    history = History.read(directory)
    done = readable_json(record, "sweep.json")
    if isinstance(done, dict) and isinstance(done.get("objective"), dict) and "value" in done["objective"]:
        reached = {"monitor": objective.get("monitor"), "value": done["objective"]["value"],
                   "turn": done["objective"].get("turn")}
    else:
        reached = reached_of(history, objective)
    state = readable_state(directory)
    return {"label": label, "dir": str(directory), "values": values, "state": state, "history": history,
            "objective": reached, "mode": objective.get("mode", "min"),
            "line": line_at(history, reached["turn"]) if len(history) else {},
            "seconds": sum(line["seconds"] for line in history if finite(line.get("seconds"))),
            "error": failure_text(record) if state == "failed" else None}


def sweep_entries(root, points, tracker):
    manifest = readable_json(Record(root), "manifest.json") or {}
    objective = settled_objective(manifest.get("objective"))
    found = {}
    for child in folders(root):
        note = readable_json(Record(child), "manifest.json") or {}
        if note.get("kind") == "point" and isinstance(note.get("id"), int):
            found[note["id"]] = (child, note)
    if points is None:
        raise ValueError(f"{root} is a sweep root; name its points with --points 3,7,12 or --points all")
    wanted = sorted(found) if points == "all" else points
    missing = [index for index in wanted if index not in found]
    if missing:
        raise ValueError(f"{root}: no point {', '.join(map(str, missing))} under the root, which holds "
                         f"{len(found)} points")
    if not wanted:
        raise ValueError(f"{root}: no point under the root yet")
    entries = []
    tracker.plan(len(wanted))
    for index in wanted:
        child, note = found[index]
        tracker.begin(f"reading point {index}")
        entry = entry_of(child, f"point {index}", note.get("values") or {}, objective)
        entries.append({**entry, "id": index})
    states = {}
    for child, _ in found.values():
        state = readable_state(child)
        states[state] = states.get(state, 0) + 1
    return manifest, objective, entries, states


def run_entries(directories, tracker):
    entries, objective = [], None
    tracker.plan(len(directories))
    for directory in directories:
        tracker.begin(f"reading {Path(directory).name}")
        record = Record(directory)
        if not record.is_record:
            raise ValueError(f"{directory} is no record directory")
        try:
            config = read_resolved(directory) or {}
        except OSError:
            config = {}
        monitor, mode = checkpoint_monitor(config)
        own = settled_objective({"monitor": monitor, "mode": mode})
        objective = objective or own
        note = readable_json(record, "manifest.json") or {}
        entries.append(entry_of(directory, Path(directory).name, note.get("params") or {}, own))
    states = {}
    for entry in entries:
        states[entry["state"]] = states.get(entry["state"], 0) + 1
    return {}, objective or settled_objective(None), entries, states


def metric_keys(entries):
    return sorted({key for entry in entries for key, value in entry["line"].items()
                   if key.startswith(("val/", "test/")) and finite(value)})


def standings(entries, keys):
    rows, wins = [], {entry["label"]: {"min": [], "max": []} for entry in entries}
    for key in keys:
        values = [(entry["label"], entry["line"][key]) for entry in entries if finite(entry["line"].get(key))]
        if not values:
            continue
        ordered = sorted(value for _, value in values)
        low, high = ordered[0], ordered[-1]
        lows = [label for label, value in values if value == low]
        highs = [label for label, value in values if value == high]
        for label in lows:
            wins[label]["min"].append(key)
        for label in highs:
            wins[label]["max"].append(key)
        rows.append({"metric": key, "points": len(values), "min": low, "min at": lows,
                     "gap to the next above": ordered[len(lows)] - low if len(ordered) > len(lows) else None,
                     "max": high, "max at": highs,
                     "gap to the next below": high - ordered[-len(highs) - 1] if len(ordered) > len(highs) else None,
                     "spread": high - low, "spread over the min": (high - low) / abs(low) if low else None})
    return rows, wins


def training_rows(entries):
    rows = []
    for entry in entries:
        history, reached = entry["history"], entry["objective"]
        turns = len(history)
        turn = reached.get("turn")
        monitor = reached.get("monitor")
        tail = [line[monitor] for line in history.lines[-5:] if monitor and finite(line.get(monitor))]
        gaps = {key[4:]: value - entry["line"][f"train/{key[4:]}"] for key, value in entry["line"].items()
                if key.startswith("val/") and finite(value) and finite(entry["line"].get(f"train/{key[4:]}"))}
        rows.append({"point": entry["label"], "state": entry["state"], "turns": turns, "objective turn": turn,
                     "turns after it": turns - turn if isinstance(turn, int) and turns else None,
                     "seconds": entry["seconds"], "seconds per turn": entry["seconds"] / turns if turns else None,
                     "spread of the last 5": statistics.pstdev(tail) if len(tail) > 1 else None, "gaps": gaps})
    return rows


def differing(entries):
    keys = sorted({key for entry in entries for key in entry["values"]})
    return keys, [key for key in keys
                  if len({json.dumps(entry["values"].get(key), sort_keys=True, default=str) for entry in entries}) > 1]


def best_entry(entries):
    scored = [entry for entry in entries if finite(entry["objective"].get("value"))]
    if not scored:
        return None
    pick = min if entries[0]["mode"] == "min" else max
    return pick(scored, key=lambda entry: entry["objective"]["value"])


def config_lines(entry):
    path = Path(entry["dir"]) / "resolved.yaml"
    return path.read_text().splitlines() if path.exists() else []


def config_diffs(entries, limit=160):
    best = best_entry(entries)
    if best is None:
        return None, []
    found = []
    for entry in entries:
        if entry is best:
            continue
        lines = list(difflib.unified_diff(config_lines(best), config_lines(entry), best["label"], entry["label"],
                                          n=1, lineterm=""))
        found.append((entry["label"], lines[:limit] + ([f"... {len(lines) - limit} more lines"]
                                                      if len(lines) > limit else [])))
    return best, found


def parquet_columns(path):
    import pyarrow.parquet

    return list(pyarrow.parquet.read_schema(path).names)


def prediction_targets(entries, name):
    pairs = []
    for entry in entries:
        path = Path(entry["dir"]) / name
        if not path.is_file():
            continue
        columns = parquet_columns(path)
        entry["columns"] = columns
        targets = [column for column in columns if not column.startswith(("pred_", "raw_")) and column != "row"]
        for pred in columns:
            truth = true_column(pred, targets) if pred.startswith("pred_") else None
            if truth is not None and (pred, truth) not in pairs:
                pairs.append((pred, truth))
    return pairs


def read_columns(entry, name, columns):
    path = Path(entry["dir"]) / name
    present = entry.get("columns") or []
    if not path.is_file() or any(column not in present for column in columns):
        return None
    return pandas.read_parquet(path, columns=list(dict.fromkeys(columns)))


def numbers_of(table, column):
    return pandas.to_numeric(table[column], errors="coerce").to_numpy(dtype="float64")


def quantiles(values, points):
    return [float(value) for value in numpy.percentile(values, points)] if len(values) else [math.nan] * len(points)


def regression_numbers(actual, guess):
    residual = guess - actual
    kept = actual != 0
    ratio = guess[kept] / actual[kept]
    low, middle, high = quantiles(ratio, [16, 50, 84])
    return {"rows": int(len(actual)), "rmse": float(numpy.sqrt(numpy.mean(residual ** 2))) if len(actual) else None,
            "mae": float(numpy.mean(numpy.abs(residual))) if len(actual) else None,
            "r2": number(r2_of(actual, guess)) if len(actual) > 1 else None,
            "bias": float(numpy.mean(residual)) if len(actual) else None,
            "median residual": float(numpy.median(residual)) if len(actual) else None,
            "response": number(middle), "resolution": number((high - low) / 2),
            "relative resolution": number((high - low) / 2 / middle) if middle else None,
            "within 10 %": float(numpy.mean(numpy.abs(ratio - 1) < 0.1)) if len(ratio) else None}


def truth_bins(actuals, count=12):
    pooled = numpy.concatenate(actuals) if actuals else numpy.zeros(0)
    edges = numpy.unique(numpy.percentile(pooled, numpy.linspace(0, 100, count + 1))) if len(pooled) else []
    return numpy.asarray(edges, dtype="float64"), pooled


def response_curves(actual, guess, edges, least=20):
    rows = []
    if len(edges) < 2:
        return rows
    index = numpy.clip(numpy.digitize(actual, edges[1:-1]), 0, len(edges) - 2)
    for position in range(len(edges) - 1):
        inside = (index == position) & (actual != 0)
        if inside.sum() < least:
            continue
        ratio = guess[inside] / actual[inside]
        low, middle, high = quantiles(ratio, [16, 50, 84])
        rows.append({"low": float(edges[position]), "high": float(edges[position + 1]),
                     "truth": float(numpy.median(actual[inside])), "rows": int(inside.sum()), "response": middle,
                     "resolution": (high - low) / 2 / middle if middle else math.nan})
    return rows


def curves_of(values, signal):
    positives, negatives = int(signal.sum()), int((~signal).sum())
    if not positives or not negatives:
        return None
    order = numpy.argsort(-values, kind="mergesort")
    ranked, hits = values[order], signal[order]
    ends = numpy.r_[numpy.flatnonzero(numpy.diff(ranked)), len(ranked) - 1]
    true_positive = numpy.cumsum(hits)[ends]
    false_positive = numpy.cumsum(~hits)[ends]
    tpr = numpy.r_[0.0, true_positive / positives]
    fpr = numpy.r_[0.0, false_positive / negatives]
    precision = numpy.r_[1.0, true_positive / (true_positive + false_positive)]
    cuts = numpy.r_[numpy.inf, ranked[ends]]
    working = []
    for efficiency in EFFICIENCIES:
        at = int(numpy.searchsorted(tpr, efficiency))
        if at < len(tpr):
            background = float(fpr[at])
            working.append({"signal": float(tpr[at]), "background": background,
                            "rejection": 1 / background if background > 0 else None, "cut": float(cuts[at])})
    return {"auc": float(numpy.sum(numpy.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2)),
            "average precision": float(numpy.sum(numpy.diff(tpr) * precision[1:])),
            "tpr": tpr, "fpr": fpr, "precision": precision, "working": working,
            "signal": positives, "background": negatives}


def calibration_of(values, signal, bins=10, least=10):
    if not len(values) or values.min() < 0 or values.max() > 1:
        return None
    edges = numpy.linspace(0, 1, bins + 1)
    index = numpy.clip(numpy.digitize(values, edges[1:-1]), 0, bins - 1)
    points = [(float(values[index == position].mean()), float(signal[index == position].mean()))
              for position in range(bins) if (index == position).sum() >= least]
    return points or None


def class_scores(truth, predicted, classes):
    texts = truth.astype(str).to_numpy()
    found = {"classes": [cell(item) for item in classes],
             "counts": [int((texts == str(item)).sum()) for item in classes]}
    if predicted is None:
        return found
    guessed = predicted.astype(str).to_numpy()
    known = {str(item) for item in classes}
    if not set(guessed[predicted.notna().to_numpy()].tolist()) <= known:
        return found
    found["accuracy"] = float((guessed == texts).mean()) if len(texts) else None
    per = []
    for item in classes:
        label = str(item)
        hit = int(((guessed == label) & (texts == label)).sum())
        picked, actual = int((guessed == label).sum()), int((texts == label).sum())
        precision = hit / picked if picked else None
        recall = hit / actual if actual else None
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
        per.append({"class": cell(item), "precision": precision, "recall": recall, "f1": f1, "rows": actual})
    found["per class"] = per
    if len(classes) <= 20:
        found["confusion"] = [[int(((texts == str(row)) & (guessed == str(column))).sum()) for column in classes]
                              for row in classes]
    return found


class Page:
    def __init__(self, title):
        self.title = title
        self.parts = []
        self.sections = []
        self.figures = 0
        self.tables = 0

    def section(self, key, title):
        self.sections.append((key, title))
        self.parts.append(f'<h2 id="{key}">{html.escape(title)}</h2>')

    def heading(self, text):
        self.parts.append(f"<h3>{html.escape(text)}</h3>")

    def text(self, text, kind=None):
        self.parts.append(f'<p class="{kind}">{html.escape(text)}</p>' if kind else f"<p>{html.escape(text)}</p>")

    def raw(self, text):
        self.parts.append(text)

    def chips(self, items):
        self.parts.append('<div class="chips">' + "".join(f'<span class="chip">{html.escape(item)}</span>'
                                                         for item in items) + "</div>")

    def table(self, headers, rows, marks=None):
        marks = marks or {}
        body = []
        for position, row in enumerate(rows):
            cells = []
            for column, value in enumerate(row):
                kind = marks.get((position, column), "")
                numeric = isinstance(value, (int, float, numpy.integer, numpy.floating)) and not isinstance(value, bool)
                classes = " ".join(item for item in ("num" if numeric else "", kind) if item)
                cells.append(f'<td class="{classes}">{html.escape(cell(value))}</td>' if classes
                             else f"<td>{html.escape(cell(value))}</td>")
            body.append("<tr>" + "".join(cells) + "</tr>")
        self.tables += 1
        self.parts.append('<div class="scroll"><table><thead><tr>' + "".join(f"<th>{html.escape(str(header))}</th>"
                                                                           for header in headers)
                          + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>")

    def figure(self, encoded, caption):
        self.figures += 1
        self.parts.append(f'<figure><img alt="{html.escape(caption)}" src="data:image/png;base64,{encoded}">'
                          f"<figcaption>{html.escape(caption)}</figcaption></figure>")

    def document(self, head):
        contents = "".join(f'<li><a href="#{key}">{html.escape(title)}</a></li>' for key, title in self.sections)
        return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{html.escape(self.title)}</title>'
                f'<meta name="viewport" content="width=device-width, initial-scale=1"><style>{CSS}</style></head>'
                f"<body><main>{head}<nav><ol>{contents}</ol></nav>{''.join(self.parts)}</main></body></html>\n")


class Drawer:
    def __init__(self):
        from .std.common.figure import Figure

        self.figures = Figure(dpi=110)
        self.pyplot = self.figures.pyplot()

    def color(self, position):
        return self.figures.categorical[position % len(self.figures.categorical)]

    def dash(self, position):
        return ("-", "--", ":", "-.")[(position // len(self.figures.categorical)) % 4]

    def encoded(self, drawing):
        buffer = io.BytesIO()
        drawing.savefig(buffer, format="png")
        self.pyplot.close(drawing)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def lines(self, series, title, xlabel, ylabel, logy=False, reference=None, marks=None):
        drawing, axis = self.figures.single(width=8.4, height=4.4)
        for position, (label, xs, ys) in enumerate(series):
            if len(xs):
                axis.plot(xs, ys, color=self.color(position), linestyle=self.dash(position), linewidth=1.6,
                          marker="o" if len(xs) <= 30 else None, markersize=3.5, label=label)
        for position, value in enumerate(marks or []):
            if value is not None:
                axis.axvline(value, color=self.color(position), linewidth=0.8, alpha=0.5, linestyle=":")
        if reference is not None:
            axis.axhline(reference, color=self.figures.ink_muted, linewidth=1, linestyle="--")
        if logy:
            axis.set_yscale("log")
        if any(len(xs) for _, xs, _ in series):
            axis.legend(loc="best", fontsize=8, ncol=2 if len(series) > 6 else 1)
        self.figures.label(axis, title, xlabel, ylabel)
        return self.encoded(drawing)

    def stairs(self, series, edges, title, xlabel, ylabel, filled=None, ratio=None):
        rows = 2 if ratio else 1
        drawing, axes = self.figures.figure(rows, 1, width=8.4, height=5.8 if ratio else 4.4,
                                            ratios=[3, 1] if ratio else None, sharex=True)
        axis = axes[0][0]
        if filled is not None:
            axis.stairs(filled[1], edges, fill=True, color=self.figures.sequential_steps[2], alpha=0.8,
                        label=filled[0])
        for position, (label, values) in enumerate(series):
            axis.stairs(values, edges, color=self.color(position), linestyle=self.dash(position), linewidth=1.5,
                        label=label)
        if series or filled is not None:
            axis.legend(loc="best", fontsize=8, ncol=2 if len(series) > 6 else 1)
        self.figures.label(axis, title, None if ratio else xlabel, ylabel)
        if ratio:
            lower = axes[1][0]
            middles = (edges[:-1] + edges[1:]) / 2
            for position, (label, values) in enumerate(ratio):
                lower.plot(middles, values, color=self.color(position), linestyle="none", marker="o", markersize=3)
            lower.axhline(1, color=self.figures.ink_muted, linewidth=1, linestyle="--")
            lower.set_ylabel("prediction / truth")
            lower.set_xlabel(xlabel)
        return self.encoded(drawing)

    def tiles(self, items, draw, title, columns=3):
        count = len(items)
        columns = min(columns, count)
        rows = math.ceil(count / columns)
        drawing, axes = self.figures.figure(rows, columns, width=4.2 * columns, height=3.7 * rows)
        for position, item in enumerate(items):
            draw(axes[position // columns][position % columns], item, drawing)
        for position in range(count, rows * columns):
            axes[position // columns][position % columns].set_visible(False)
        self.figures.title(drawing, title)
        return self.encoded(drawing)


def spread_edges(arrays, bins=50):
    pooled = numpy.concatenate([array for array in arrays if len(array)]) if any(len(array) for array in arrays) \
        else numpy.zeros(0)
    if not len(pooled):
        return None
    low, high = numpy.percentile(pooled, [0.5, 99.5])
    if high <= low:
        high = low + 1
    return numpy.linspace(low, high, bins + 1)


def fractions(values, edges):
    counts, _ = numpy.histogram(values, bins=edges)
    return counts / len(values) if len(values) else counts.astype("float64")


def regression_section(page, drawer, entries, pred, truth, name, files, out, tracker):
    rows, loaded = [], []
    for entry in entries:
        tracker.begin(f"{truth} as a regression: {entry['label']}")
        column = pred if pred in (entry.get("columns") or []) else None
        table = read_columns(entry, name, [truth, column]) if column else None
        if table is None:
            continue
        actual, guess = numbers_of(table, truth), numbers_of(table, column)
        kept = numpy.isfinite(actual) & numpy.isfinite(guess)
        actual, guess = actual[kept], guess[kept]
        if not len(actual):
            continue
        loaded.append((entry["label"], actual, guess))
        rows.append({"point": entry["label"], **regression_numbers(actual, guess)})
    tracker.begin(f"{truth} as a regression: the figures")
    if not loaded:
        page.text(f"No selected point holds {pred} and {truth} in {name}.", "note")
        return {"target": truth, "prediction": pred, "task": "regression", "points": []}
    headers = ["point", "rows", *REGRESSION_BEST]
    marks = {}
    for column, key in enumerate(REGRESSION_BEST, start=2):
        values = [(position, row[key]) for position, row in enumerate(rows) if row[key] is not None]
        if len(values) > 1:
            marks[(min(values, key=lambda item: REGRESSION_BEST[key](item[1]))[0], column)] = "low"
    page.table(headers, [[row[key] for key in headers] for row in rows], marks)
    page.text("Marked: the lowest rmse, mae and resolution, the highest r2 and share within 10 %, the response "
              "closest to 1 and the bias and median residual closest to 0.", "muted")
    csv = out / f"regression_{safe(truth)}_{safe(pred)}.csv"
    pandas.DataFrame(rows).to_csv(csv, index=False)
    files.append(csv.name)
    edges = spread_edges([numpy.concatenate([actual, guess]) for _, actual, guess in loaded])
    same = all(len(actual) == len(loaded[0][1]) and numpy.array_equal(actual, loaded[0][1])
               for _, actual, _ in loaded)
    if edges is not None:
        truths = [(f"{truth} ({label})", fractions(actual, edges)) for label, actual, _ in loaded]
        guesses = [(label, fractions(guess, edges)) for label, _, guess in loaded]
        ratios = []
        for label, actual, guess in loaded:
            counts_true, _ = numpy.histogram(actual, bins=edges)
            counts_pred, _ = numpy.histogram(guess, bins=edges)
            with numpy.errstate(divide="ignore", invalid="ignore"):
                ratios.append((label, numpy.where(counts_true > 0, counts_pred / numpy.maximum(counts_true, 1),
                                                  numpy.nan)))
        page.figure(drawer.stairs(guesses if same else truths + guesses, edges, f"{truth}: the predictions over the "
                                  f"truth", truth, "fraction of rows",
                                  filled=(f"{truth} (truth)", fractions(loaded[0][1], edges)) if same else None,
                                  ratio=ratios),
                    f"The distribution of every selected point's {pred} over the same bins as the truth, as the "
                    f"fraction of its rows, and below the ratio of the counts per bin"
                    + ("" if same else "; the points were tested on different rows, so every truth is drawn") + ".")
    residuals = [(label, guess - actual) for label, actual, guess in loaded]
    edges = spread_edges([values for _, values in residuals])
    if edges is not None:
        page.figure(drawer.stairs([(label, fractions(values, edges)) for label, values in residuals], edges,
                                  f"{truth}: residual", f"{pred} minus {truth}", "fraction of rows"),
                    "The residual, prediction minus truth, of every selected point.")
    relative = [(label, guess[actual != 0] / actual[actual != 0] - 1) for label, actual, guess in loaded]
    edges = spread_edges([values for _, values in relative])
    if edges is not None:
        page.figure(drawer.stairs([(label, fractions(values, edges)) for label, values in relative], edges,
                                  f"{truth}: relative residual", "prediction / truth - 1", "fraction of rows"),
                    "The relative residual, prediction over truth minus one, on the rows whose truth is not 0.")
    bins, _ = truth_bins([actual for _, actual, _ in loaded])
    profiles = [(label, response_curves(actual, guess, bins)) for label, actual, guess in loaded]
    if any(profile for _, profile in profiles):
        page.figure(drawer.lines([(label, [row["truth"] for row in profile], [row["response"] for row in profile])
                                  for label, profile in profiles], f"{truth}: response", f"{truth} (median in the bin)",
                                 "median of prediction / truth", reference=1.0),
                    "The response: the median of prediction over truth in 12 bins of the truth that hold the same "
                    "number of rows; 1 is an unbiased prediction.")
        page.figure(drawer.lines([(label, [row["truth"] for row in profile],
                                   [row["resolution"] for row in profile]) for label, profile in profiles],
                                 f"{truth}: resolution", f"{truth} (median in the bin)",
                                 "half the 16 to 84 % width over the median"),
                    "The relative resolution in the same bins: half the width between the 16th and the 84th "
                    "percentile of prediction over truth, over its median; lower is sharper.")

    def density(axis, item, drawing):
        label, actual, guess = item
        drawer.figures.density(drawing, axis, actual, guess, gridsize=45)
        low = float(min(actual.min(), guess.min()))
        high = float(max(actual.max(), guess.max()))
        axis.plot([low, high], [low, high], color=drawer.figures.ink_muted, linewidth=1, linestyle="--")
        drawer.figures.label(axis, label, truth, pred)

    page.figure(drawer.tiles(loaded, density, f"{pred} against {truth}"),
                "The prediction against the truth for every selected point, as the log of the rows per cell, with "
                "the y = x line.")
    return {"target": truth, "prediction": pred, "task": "regression", "points": rows,
            "response": {label: profile for label, profile in profiles}}


def classification_section(page, drawer, entries, pred, truth, name, score, signal, files, out, tracker):
    rows, loaded = [], []
    for position, entry in enumerate(entries):
        tracker.begin(f"{truth} as a classification: {entry['label']}")
        columns = [truth] + ([pred] if pred in (entry.get("columns") or []) else []) \
            + ([score] if score and score in (entry.get("columns") or []) else [])
        table = read_columns(entry, name, columns)
        if table is None or truth not in table.columns:
            continue
        labels = table[truth]
        keep = labels.notna().to_numpy()
        table = table[keep]
        classes = classes_of(table[truth])
        if len(classes) > 100:
            page.text(f"{truth} holds {len(classes)} distinct values in {entry['label']}; a classification takes at "
                      f"most 100 classes, so {truth} is no classification target.", "note")
            tracker.plan(tracker.total - (len(entries) - position - 1))
            tracker.begin(f"{truth} as a classification: stopped")
            return {"target": truth, "prediction": pred, "task": "classification", "points": []}
        numbers = class_scores(table[truth], table[pred] if pred in table.columns else None, classes)
        found = {"point": entry["label"], **numbers}
        if score and score in table.columns and signal is not None:
            matched = [item for item in classes if same_label(item, signal)]
            values = numbers_of(table, score)
            usable = numpy.isfinite(values)
            if matched:
                texts = table[truth].astype(str).to_numpy()
                curves = curves_of(values[usable], texts[usable] == str(matched[0]))
                found["curves"] = curves
                found["calibration"] = calibration_of(values[usable], texts[usable] == str(matched[0]))
                found["scores"] = {str(item): values[usable][texts[usable] == str(item)] for item in classes[:10]}
        loaded.append(found)
    tracker.begin(f"{truth} as a classification: the figures")
    if not loaded:
        page.text(f"No selected point holds {truth} in {name}.", "note")
        return {"target": truth, "prediction": pred, "task": "classification", "points": []}
    classes = loaded[0]["classes"]
    page.chips([f"{item}: {count} rows in {loaded[0]['point']}" for item, count in zip(classes, loaded[0]["counts"])])
    headers = ["point", "accuracy", "macro f1", "auc", "average precision",
               *[f"rejection at {int(level * 100)} %" for level in EFFICIENCIES]]
    table_rows, marks = [], {}
    for found in loaded:
        per = found.get("per class") or []
        f1s = [item["f1"] for item in per if item["f1"] is not None]
        curves = found.get("curves")
        rejections = []
        for level in EFFICIENCIES:
            match = next((item for item in (curves or {}).get("working", []) if item["signal"] >= level), None)
            rejections.append(match["rejection"] if match else None)
        table_rows.append([found["point"], found.get("accuracy"), statistics.fmean(f1s) if f1s else None,
                           curves["auc"] if curves else None, curves["average precision"] if curves else None,
                           *rejections])
    for column in range(1, len(headers)):
        values = [(position, row[column]) for position, row in enumerate(table_rows) if row[column] is not None]
        if len(values) > 1:
            marks[(max(values, key=lambda item: item[1])[0], column)] = "low"
    page.table(headers, table_rows, marks)
    page.text("Marked: the highest value of every column. The rejection is one over the background efficiency at "
              "the first cut that keeps at least that signal efficiency.", "muted")
    csv = out / f"classification_{safe(truth)}_{safe(pred)}.csv"
    pandas.DataFrame([dict(zip(headers, row)) for row in table_rows]).to_csv(csv, index=False)
    files.append(csv.name)
    per_rows = [[found["point"], item["class"], item["rows"], item["precision"], item["recall"], item["f1"]]
                for found in loaded for item in found.get("per class") or []]
    if per_rows:
        page.heading(f"{truth}: precision, recall and F1 per class")
        page.table(["point", "class", "rows", "precision", "recall", "f1"], per_rows)
    with_curves = [found for found in loaded if found.get("curves")]
    if not with_curves:
        page.text(f"Give --score {truth}=COLUMN and --signal {truth}=CLASS for the ROC, the precision and recall "
                  f"curve, the score distributions and the calibration.", "note")
    else:
        page.figure(drawer.lines([(found["point"], found["curves"]["fpr"], found["curves"]["tpr"])
                                  for found in with_curves], f"{truth}: ROC of {score}, {signal} as signal",
                                 "background efficiency", "signal efficiency"),
                    "The ROC: the signal efficiency against the background efficiency over every cut on the score.")
        rejection = []
        for found in with_curves:
            curves = found["curves"]
            usable = curves["fpr"] > 0
            rejection.append((found["point"], curves["tpr"][usable], 1 / curves["fpr"][usable]))
        page.figure(drawer.lines(rejection, f"{truth}: background rejection", "signal efficiency",
                                 "1 / background efficiency", logy=True),
                    "The background rejection against the signal efficiency, on a log axis.")
        page.figure(drawer.lines([(found["point"], found["curves"]["tpr"], found["curves"]["precision"])
                                  for found in with_curves], f"{truth}: precision and recall", "recall",
                                 "precision"),
                    "The precision against the recall of the signal class over every cut on the score.")
        calibrated = [found for found in with_curves if found.get("calibration")]
        if calibrated:
            page.figure(drawer.lines([(found["point"], [point[0] for point in found["calibration"]],
                                       [point[1] for point in found["calibration"]]) for found in calibrated],
                                     f"{truth}: calibration", f"mean {score} in the bin", f"share of {signal}"),
                        "The calibration: the share of the signal class against the mean score in ten bins of "
                        "the score between 0 and 1; a calibrated probability follows y = x.")
        edges = spread_edges([values for found in with_curves for values in found["scores"].values()])
        if edges is not None:
            def scores(axis, found, drawing):
                for position, (label, values) in enumerate(found["scores"].items()):
                    axis.stairs(fractions(values, edges), edges, color=drawer.color(position), linewidth=1.4,
                                label=label)
                axis.legend(loc="best", fontsize=8)
                drawer.figures.label(axis, found["point"], score, "fraction of the class")

            page.figure(drawer.tiles(with_curves, scores, f"{score} by class of {truth}"),
                        "The score distribution of every class, as the fraction of the rows of the class.")
    confused = [found for found in loaded if found.get("confusion")]
    if confused:
        def confusion(axis, found, drawing):
            matrix = numpy.asarray(found["confusion"], dtype="float64")
            shares = matrix / numpy.maximum(matrix.sum(axis=1, keepdims=True), 1)
            axis.imshow(shares, cmap=drawer.figures.sequential(), vmin=0, vmax=1)
            ticks = range(len(found["classes"]))
            axis.set_xticks(ticks, found["classes"], rotation=45, ha="right", fontsize=7)
            axis.set_yticks(ticks, found["classes"], fontsize=7)
            axis.grid(False)
            if len(ticks) <= 8:
                for row in ticks:
                    for column in ticks:
                        axis.text(column, row, f"{shares[row, column]:.2f}", ha="center", va="center", fontsize=7,
                                  color="white" if shares[row, column] > 0.5 else drawer.figures.ink)
            drawer.figures.label(axis, found["point"], f"predicted {truth}", f"true {truth}")

        page.figure(drawer.tiles(confused, confusion, f"{truth}: confusion, as the share of every true class"),
                    "The confusion of the predicted classes, every row divided by the rows of its true class.")
    keep = [{key: value for key, value in found.items() if key not in ("curves", "scores")}
            | ({"auc": found["curves"]["auc"], "average precision": found["curves"]["average precision"],
                "working": found["curves"]["working"]} if found.get("curves") else {}) for found in loaded]
    return {"target": truth, "prediction": pred, "task": "classification", "score": score, "signal": signal,
            "points": keep}


def safe(text):
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in str(text))


def default_out(records):
    first = Path(records[0])
    return (first if len(records) == 1 else first.parent) / "reports" / REPORT


def clear_previous(out):
    previous = out / "report.json"
    if not previous.is_file():
        return
    try:
        written = json.loads(previous.read_text()).get("files") or []
    except (ValueError, OSError):
        written = []
    for name in written:
        target = out / Path(name).name
        if target.is_file():
            target.unlink()


def plain(value):
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, numpy.ndarray):
        return plain(value.tolist())
    if isinstance(value, (numpy.floating, numpy.integer)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def report(records, points=None, tasks=None, scores=None, signals=None, predictions="predictions.parquet", out=None,
           progress=None):
    records = [Path(record) for record in records]
    tracker = Progress(progress)
    tasks, scores, signals = dict(tasks or {}), dict(scores or {}), dict(signals or {})
    wrong = [f"{target}={task}" for target, task in tasks.items() if task not in TASKS]
    if wrong:
        raise ValueError(f"--task takes regression or classification, got {', '.join(wrong)}")
    note = readable_json(Record(records[0]), "manifest.json") or {}
    sweep = len(records) == 1 and note.get("kind") == "sweep"
    if sweep:
        manifest, objective, entries, states = sweep_entries(records[0], points, tracker)
    else:
        if points is not None:
            raise ValueError("--points picks the points of one sweep root; give record directories without it")
        manifest, objective, entries, states = run_entries(records, tracker)
    pairs = prediction_targets(entries, predictions)
    series = sorted({key for entry in entries for line in entry["history"].lines for key, value in line.items()
                     if key.startswith(("val/", "train/", "test/")) and finite(value)})
    tracker.plan(tracker.done + len(series) + len(pairs) * (len(entries) + 1) + 1)
    target = Path(out) if out is not None else default_out(records)
    target.mkdir(parents=True, exist_ok=True)
    clear_previous(target)
    files = []
    drawer = Drawer()
    name = records[0].name if sweep else ", ".join(record.name for record in records)
    page = Page(f"kalfa report: {name}")
    monitor = objective.get("monitor")
    head = (f"<h1>kalfa report: {html.escape(name)}</h1><p class=\"muted\">written "
            f"{datetime.now().isoformat(timespec='seconds')} by kalfa {html.escape(str(__version__))} from "
            f"{html.escape(str(records[0].resolve()) if sweep else ', '.join(str(record) for record in records))}"
            f"</p>")
    page.section("summary", "Summary")
    if sweep:
        page.chips([f"objective {monitor} ({objective['mode']}, {objective['at']})",
                    f"strategy {manifest.get('strategy') or 'unknown'}",
                    f"{manifest.get('total') or sum(states.values())} points planned",
                    *[f"{count} {state}" for state, count in sorted(states.items())]])
    else:
        page.chips([f"objective {monitor or 'the last turn'}",
                    *[f"{count} {state}" for state, count in states.items()]])
    total = sum(entry["seconds"] for entry in entries)
    page.text(f"{len(entries)} selected: {', '.join(entry['label'] for entry in entries)}. Their training took "
              f"{total / 3600:.2f} hours in all.")
    best = best_entry(entries)
    if best is not None:
        page.text(f"The best selected by {monitor} is {best['label']}: {cell(best['objective']['value'])} at turn "
                  f"{best['objective']['turn']}.")
    keys, varying = differing(entries)
    overview = [[entry["label"], entry["state"], entry["objective"].get("value"), entry["objective"].get("turn"),
                 len(entry["history"]), entry["seconds"], *[entry["values"].get(key) for key in varying],
                 f"{Path(entry['dir']).name}"] for entry in entries]
    page.table(["point", "state", monitor or "objective", "turn", "turns", "seconds", *varying, "record"], overview)
    pandas.DataFrame([dict(zip(["point", "state", "objective", "turn", "turns", "seconds", *varying, "record"], row))
                      for row in overview]).to_csv(target / "points.csv", index=False)
    files.append("points.csv")
    metrics = metric_keys(entries)
    rows, wins = standings(entries, metrics)
    page.section("metrics", "Every metric at the objective turn")
    if metrics:
        marks = {}
        for column, key in enumerate(metrics, start=1):
            values = [(position, entry["line"].get(key)) for position, entry in enumerate(entries)
                      if finite(entry["line"].get(key))]
            if len(values) > 1:
                low = min(value for _, value in values)
                high = max(value for _, value in values)
                for position, value in values:
                    if value == low and value == high:
                        marks[(position, column)] = "both"
                    elif value == low:
                        marks[(position, column)] = "low"
                    elif value == high:
                        marks[(position, column)] = "high"
        page.table(["point", *metrics], [[entry["label"], *[entry["line"].get(key) for key in metrics]]
                                          for entry in entries], marks)
        page.text("Green is the minimum of a column, red its maximum; which one is better depends on the metric, so "
                  "the report names both and judges neither.", "muted")
        pandas.DataFrame([{"point": entry["label"], **{key: entry["line"].get(key) for key in metrics}}
                          for entry in entries]).to_csv(target / "metrics.csv", index=False)
        files.append("metrics.csv")
        page.section("standings", "Who takes what")
        page.table(["point", "minimum in", "maximum in", "metrics with the minimum", "metrics with the maximum"],
                   sorted([[label, len(found["min"]), len(found["max"]), ", ".join(found["min"]),
                            ", ".join(found["max"])] for label, found in wins.items()],
                          key=lambda row: (-row[1], row[2])))
        page.text(f"Out of {len(metrics)} metrics; a tie counts for every point that reaches the value.", "muted")
        page.table(["metric", "min", "min at", "gap to the next above", "max", "max at", "gap to the next below",
                    "spread", "spread over the min"],
                   [[row["metric"], row["min"], ", ".join(row["min at"]), row["gap to the next above"], row["max"],
                     ", ".join(row["max at"]), row["gap to the next below"], row["spread"],
                     row["spread over the min"]] for row in rows])
    else:
        page.text("No val/ or test/ value at the objective turn of any selected point.", "note")
    page.section("training", "Training")
    training = training_rows(entries)
    gap_keys = sorted({key for row in training for key in row["gaps"]})
    page.table(["point", "state", "turns", "objective turn", "turns after it", "seconds", "seconds per turn",
                "spread of the last 5", *[f"val - train {key}" for key in gap_keys]],
               [[row["point"], row["state"], row["turns"], row["objective turn"], row["turns after it"],
                 row["seconds"], row["seconds per turn"], row["spread of the last 5"],
                 *[row["gaps"].get(key) for key in gap_keys]] for row in training])
    page.text(f"The spread of the last 5 is the standard deviation of {monitor or 'the objective'} over the last five "
              f"turns; val - train is the value on the validation set minus the one on the training set at the "
              f"objective turn, a first look at over fitting.", "muted")
    for key in series:
        tracker.begin(f"training: {key}")
        drawn = [(entry["label"], [line["turn"] for line in entry["history"].lines if finite(line.get(key))],
                  [line[key] for line in entry["history"].lines if finite(line.get(key))]) for entry in entries]
        if sum(len(xs) for _, xs, _ in drawn) < 2:
            continue
        page.figure(drawer.lines(drawn, key, "turn", key, marks=[entry["objective"].get("turn") for entry in entries]),
                    f"{key} over the turns; a dotted line marks the objective turn of the point with the same colour.")
    page.section("params", "Params")
    if varying:
        page.table(["point", *varying], [[entry["label"], *[entry["values"].get(key) for key in varying]]
                                         for entry in entries])
    same = [key for key in keys if key not in varying]
    if same:
        page.text("The same in every selected point: " + ", ".join(
            f"{key} = {cell(entries[0]['values'].get(key))}" for key in same) + ".", "muted")
    best_config, diffs = config_diffs(entries)
    for label, lines in diffs:
        page.raw(f"<details><summary>resolved.yaml, {html.escape(best_config['label'])} against "
                 f"{html.escape(label)}</summary><pre>{html.escape(chr(10).join(lines) or 'no difference')}</pre>"
                 f"</details>")
    sections = []
    for pred, truth in pairs:
        task = tasks.get(truth, "regression")
        key = f"target-{safe(truth)}-{safe(pred)}"
        page.section(key, f"{truth} as a {task} ({pred})")
        if truth not in tasks:
            page.text(f"{truth} is reported as a regression; --task {truth}=classification reports it as classes.",
                      "muted")
        if task == "regression":
            sections.append(regression_section(page, drawer, entries, pred, truth, predictions, files, target,
                                               tracker))
        else:
            sections.append(classification_section(page, drawer, entries, pred, truth, predictions,
                                                   scores.get(truth), signals.get(truth), files, target, tracker))
    if not pairs:
        page.section("predictions", "Predictions")
        page.text(f"No selected point wrote {predictions} with a pred_ column next to its target.", "note")
    failed = [entry for entry in entries if entry["state"] in ("failed", "lost", "unreadable")]
    if failed:
        page.section("failures", "Failed and lost")
        page.table(["point", "state", "reason"], [[entry["label"], entry["state"], entry["error"] or ""]
                                                  for entry in failed])
    page.section("terms", "What the numbers mean")
    page.raw("<dl class=\"terms\">"
             "<dt>objective turn</dt><dd>the turn the sweep objective, or the checkpoint monitor of a run, was best "
             "at; every metric is read at that turn</dd>"
             "<dt>response</dt><dd>the median of prediction over truth, on the rows whose truth is not 0</dd>"
             "<dt>resolution</dt><dd>half the width between the 16th and the 84th percentile of prediction over "
             "truth; the relative resolution divides it by the response</dd>"
             "<dt>within 10 %</dt><dd>the share of rows whose prediction is within 10 % of the truth</dd>"
             "<dt>bias</dt><dd>the mean of prediction minus truth</dd>"
             "<dt>rejection</dt><dd>one over the background efficiency at the first cut on the score that keeps at "
             "least the signal efficiency named</dd>"
             "<dt>average precision</dt><dd>the area under the precision and recall curve, summed over the cuts</dd>"
             "</dl>")
    tracker.begin("writing report.html")
    (target / "report.html").write_text(page.document(head))
    files.append("report.html")
    summary = {"records": [str(record) for record in records], "points": [entry["label"] for entry in entries],
               "objective": objective, "written": datetime.now().isoformat(timespec="seconds"),
               "standings": {"metrics": rows, "wins": wins}, "training": training, "targets": sections,
               "files": [*files, "report.json"]}
    (target / "report.json").write_text(json.dumps(plain(summary), indent=2, default=str))
    return str(target), page.figures, page.tables, [*files, "report.json"]
