import json
import math
import re
import textwrap
from pathlib import Path

import pandas

from .describe.text import PLAIN, field_line, head, table, width_of
from .record import Record, read_resolved
from .std.common.history import History


NUMERIC = re.compile(r"^-?(\d[\d.eE+\-/]*|nan|inf)$")


def cell(value):
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def align_numbers(headers, rows):
    for position in range(len(headers)):
        column = [row[position] for row in rows if row[position]]
        if not column or not all(NUMERIC.match(value) for value in column):
            continue
        size = max(len(headers[position]), *(len(value) for value in column))
        headers[position] = headers[position].rjust(size)
        for row in rows:
            row[position] = row[position].rjust(size)
    return headers, rows


def numbered(headers, rows, style, width):
    return table(*align_numbers(headers, rows), style, width=width)


def swept_of(row):
    return [key for key in row if key not in ("id", "objective", "turn", "turns", "dir") and "/" not in key]


def metrics_of(rows):
    return sorted({key for row in rows for key in row if "/" in key})


def wrapped(label, text, style, width, label_width=11):
    room = max(24, width - 2 - label_width)
    pieces = textwrap.wrap(text, room, break_on_hyphens=False) or [""]
    return [field_line(label if not position else "", piece, style, label_width=label_width)
            for position, piece in enumerate(pieces)]


def block(title, width, style):
    return [head(title, width, style), ""]


def mean_std(values):
    clean = [float(value) for value in values if not (isinstance(value, float) and math.isnan(value))]
    if not clean:
        return math.nan, math.nan
    mean = sum(clean) / len(clean)
    if len(clean) < 2:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in clean) / (len(clean) - 1)
    return mean, math.sqrt(variance)


def load_runs(run_dirs):
    runs = []
    for directory in run_dirs:
        directory = Path(directory)
        if not (directory / "resolved.yaml").exists():
            continue
        runs.append({"dir": str(directory), "config": read_resolved(directory), "history": History.read(directory)})
    return runs


def fold_summary(runs):
    folds = []
    for entry in runs:
        params = entry["config"].get("params") or {}
        folds.append({"dir": entry["dir"], "fold": params.get("fold"), "turns": len(entry["history"]),
                      "last": History(entry["history"]).last()})
    folds.sort(key=lambda item: (item["fold"] is None, item["fold"]))
    keys = sorted({key for fold in folds for key in fold["last"]})
    summary = {}
    for key in keys:
        mean, std = mean_std([fold["last"].get(key, math.nan) for fold in folds])
        summary[key] = {"mean": mean, "std": std}
    return {"folds": folds, "summary": summary}


def fold_markdown(result):
    keys = list(result["summary"])
    lines = ["# k fold summary", "", "| metric | mean | std |", "|---|---|---|"]
    for key in keys:
        entry = result["summary"][key]
        lines.append(f"| {key} | {cell(entry['mean'])} | {cell(entry['std'])} |")
    lines += ["", "| fold | turns | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
    for fold in result["folds"]:
        values = " | ".join(cell(fold["last"].get(key, math.nan)) for key in keys)
        lines.append(f"| {fold['fold']} | {fold['turns']} | {values} |")
    return "\n".join(lines) + "\n"


def fold_text(result, style=PLAIN, width=None):
    width = width or width_of()
    summary = result["summary"]
    lines = block("CROSS VALIDATION", width, style)
    lines += wrapped("folds", f"{len(result['folds'])}: "
                              + ", ".join(str(fold["fold"]) for fold in result["folds"]), style, width)
    lines.append("")
    rows = [[key, cell(summary[key]["mean"]), cell(summary[key]["std"])] for key in summary]
    lines += numbered(["metric", "mean", "std"], rows, style, width)
    lines += ["", *wrapped("per fold", "cv.md carries the value every fold reached", style, width), ""]
    return "\n".join(lines) + "\n"


def sweep_table(runs):
    params = [entry["config"].get("params") or {} for entry in runs]
    keys = sorted({key for values in params for key in values})
    varying = [key for key in keys if len({json.dumps(values.get(key), sort_keys=True) for values in params}) > 1]
    rows = []
    for entry, values in zip(runs, params):
        row = {"dir": entry["dir"], "turns": len(entry["history"])}
        for key in varying:
            row[key] = values.get(key)
        row.update(History(entry["history"]).last(("val/", "test/")))
        rows.append(row)
    return {"varying": varying, "rows": rows}


def sweep_columns(result):
    return ["dir", "turns", *result["varying"], *metrics_of(result["rows"])]


def sweep_markdown(result):
    columns = sweep_columns(result)
    lines = ["# sweep", "", "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in result["rows"]:
        lines.append("| " + " | ".join(cell(row.get(column)) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def sweep_text(result, style=PLAIN, width=None):
    width = width or width_of()
    columns = sweep_columns(result)
    rows = [[cell(row.get(column)) for column in columns] for row in result["rows"]]
    lines = block("RUNS", width, style)
    lines += numbered(columns, rows, style, width)
    lines.append("")
    return "\n".join(lines) + "\n"


def is_run_dir(path):
    record = Record(path)
    manifest = record.read_json("manifest.json")
    if manifest is not None:
        return manifest.get("kind") in ("run", "point")
    return record.path("resolved.yaml").exists()


def collect_root(root, out=None, markdown=False, style=PLAIN):
    root = Path(root)
    finished = []
    skipped = []
    for child in sorted(path for path in root.iterdir() if path.is_dir()):
        info = child / "sweep.json"
        if info.exists():
            point = json.loads(info.read_text())
            history = History.read(child)
            finished.append({**point, "dir": str(child), "turns": len(history),
                             "last": history.last(("val/", "test/"))})
        elif is_run_dir(child) or (child / "run.json").exists():
            status = "unfinished"
            summary = child / "run.json"
            if summary.exists():
                try:
                    status = "failed" if json.loads(summary.read_text()).get("status") != "ok" else "unfinished"
                except ValueError:
                    status = "failed"
            skipped.append({"dir": child.name, "status": status})
    if not finished:
        raise ValueError(f"{root}: no finished point (a directory with sweep.json) under the sweep root")
    manifest = Record(root).read_json("manifest.json") or {}
    objective = {**finished[0]["objective"], **(manifest.get("objective") or {})}
    pick = min if objective.get("mode", "min") == "min" else max
    best = pick(finished, key=lambda entry: entry["objective"]["value"])
    rows = []
    for entry in finished:
        row = {"id": entry["id"], **entry["point"], "objective": entry["objective"]["value"],
               "turn": entry["objective"]["turn"], "turns": entry["turns"], **entry["last"]}
        rows.append(row)
    result = {"objective": {key: objective.get(key) for key in ("monitor", "mode", "at")}, "points": rows,
              "best": {"id": best["id"], "point": best["point"], "value": best["objective"]["value"],
                       "turn": best["objective"]["turn"], "dir": best["dir"]},
              "skipped": skipped}
    target = Path(out) if out is not None else root
    target.mkdir(parents=True, exist_ok=True)
    (target / "sweep.json").write_text(json.dumps(result, indent=2, default=float))

    pandas.DataFrame(rows).to_csv(target / "sweep.csv", index=False)
    report = root_markdown(result)
    (target / "sweep.md").write_text(report)
    return "sweep", report if markdown else root_text(result, style), str(target)


def skipped_by_status(result):
    grouped = []
    for status in ("failed", "unfinished"):
        names = [entry["dir"] for entry in result["skipped"] if entry["status"] == status]
        if names:
            grouped.append((status, names))
    return grouped


def best_point(best):
    return ", ".join(f"{name}={cell(value)}" for name, value in best["point"].items())


def root_markdown(result):
    rows, objective, best = result["points"], result["objective"], result["best"]
    columns = ["id", *swept_of(rows[0]), "objective", "turn", "turns", *metrics_of(rows)]
    counted = len(rows) + len(result["skipped"])
    lines = [f"# sweep: {objective['monitor']} ({objective['mode']}, {objective['at']})", "",
             f"{len(rows)} of {counted} points finished.", "",
             "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join(cell(row.get(column)) for column in columns) + " |")
    lines += ["", f"best: point {best['id']} with {objective['monitor']}={cell(best['value'])} at turn "
                  f"{best['turn']} ({best['dir']}), {best_point(best)}"]
    for status, names in skipped_by_status(result):
        lines.append(f"{status}: {len(names)} ({', '.join(names)})")
    return "\n".join(lines) + "\n"


def root_text(result, style=PLAIN, width=None):
    width = width or width_of()
    rows, objective, best = result["points"], result["objective"], result["best"]
    swept = swept_of(rows[0])
    body = [["*" if row["id"] == best["id"] else "",
             *[cell(row.get(key)) for key in ("id", *swept, "objective")],
             f"{row['turn']}/{row['turns']}"] for row in rows]
    counted = len(rows) + len(result["skipped"])
    lines = block("SWEEP", width, style)
    lines += wrapped("objective", f"{objective['monitor']} ({objective['mode']}, {objective['at']})", style, width)
    lines += wrapped("points", f"{len(rows)} of {counted} finished", style, width)
    lines.append("")
    lines += numbered(["", "id", *swept, "objective", "turn"], body, style, width)
    lines += ["", *wrapped("best", f"point {best['id']}, {objective['monitor']} = {cell(best['value'])} at turn "
                                   f"{best['turn']}, {best['dir']}", style, width)]
    lines += wrapped("", best_point(best), style, width)
    for status, names in skipped_by_status(result):
        lines += wrapped(status, f"{len(names)}: " + ", ".join(names), style, width)
    metrics = metrics_of(rows)
    if metrics:
        lines += wrapped("metrics", f"{len(metrics)} per point, in sweep.csv and sweep.md, or here with "
                                    f"--markdown", style, width)
    lines.append("")
    return "\n".join(lines) + "\n"


def collect(run_dirs, out=None, markdown=False, style=PLAIN):
    if len(run_dirs) == 1 and Path(run_dirs[0]).is_dir() and not is_run_dir(run_dirs[0]):
        return collect_root(run_dirs[0], out, markdown, style)
    runs = load_runs(run_dirs)
    if not runs:
        raise ValueError("no run with a resolved.yaml among the given directories")
    target = Path(out) if out is not None else Path(runs[0]["dir"]).parent
    target.mkdir(parents=True, exist_ok=True)
    if all((entry["config"].get("params") or {}).get("fold") is not None for entry in runs):
        result = fold_summary(runs)
        (target / "cv.json").write_text(json.dumps(result, indent=2, default=float))
        report = fold_markdown(result)
        (target / "cv.md").write_text(report)
        return "cv", report if markdown else fold_text(result, style), str(target)
    result = sweep_table(runs)
    (target / "sweep.json").write_text(json.dumps(result, indent=2, default=float))
    pandas.DataFrame(result["rows"], columns=sweep_columns(result)).to_csv(target / "sweep.csv", index=False)
    report = sweep_markdown(result)
    (target / "sweep.md").write_text(report)
    return "sweep", report if markdown else sweep_text(result, style), str(target)
