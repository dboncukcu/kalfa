"""kalfa collect: the k fold summary of fold runs, or the sweep table of a parameter scan."""

import json
import math
from pathlib import Path

from .record import read_history, read_resolved


def _last_values(history, prefixes=("test/",)):
    if not history:
        return {}
    last = history[-1]
    return {key: value for key, value in last.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool) and key.startswith(prefixes)}


def _mean_std(values):
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
        runs.append({"dir": str(directory), "config": read_resolved(directory), "history": read_history(directory)})
    return runs


def fold_summary(runs):
    folds = []
    for entry in runs:
        params = entry["config"].get("params") or {}
        folds.append({"dir": entry["dir"], "fold": params.get("fold"), "turns": len(entry["history"]),
                      "last": _last_values(entry["history"])})
    folds.sort(key=lambda item: (item["fold"] is None, item["fold"]))
    keys = sorted({key for fold in folds for key in fold["last"]})
    summary = {}
    for key in keys:
        mean, std = _mean_std([fold["last"].get(key, math.nan) for fold in folds])
        summary[key] = {"mean": mean, "std": std}
    return {"folds": folds, "summary": summary}


def fold_markdown(result):
    keys = list(result["summary"])
    lines = ["# k fold summary", "", "| metric | mean | std |", "|---|---|---|"]
    for key in keys:
        entry = result["summary"][key]
        lines.append(f"| {key} | {entry['mean']:.6g} | {entry['std']:.6g} |")
    lines += ["", "| fold | turns | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
    for fold in result["folds"]:
        values = " | ".join(f"{fold['last'].get(key, math.nan):.6g}" for key in keys)
        lines.append(f"| {fold['fold']} | {fold['turns']} | {values} |")
    return "\n".join(lines) + "\n"


def sweep_table(runs):
    params = [entry["config"].get("params") or {} for entry in runs]
    keys = sorted({key for table in params for key in table})
    varying = [key for key in keys if len({json.dumps(table.get(key), sort_keys=True) for table in params}) > 1]
    rows = []
    for entry, table in zip(runs, params):
        row = {"dir": entry["dir"], "turns": len(entry["history"])}
        for key in varying:
            row[key] = table.get(key)
        row.update(_last_values(entry["history"], ("val/", "test/")))
        rows.append(row)
    return {"varying": varying, "rows": rows}


def sweep_markdown(result):
    columns = ["dir", "turns", *result["varying"]]
    metrics = sorted({key for row in result["rows"] for key in row if key.startswith(("val/", "test/"))})
    columns += metrics
    lines = ["# sweep", "", "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in result["rows"]:
        cells = []
        for column in columns:
            value = row.get(column)
            cells.append(f"{value:.6g}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def is_run_dir(path):
    return (Path(path) / "resolved.yaml").exists()


def collect_root(root, out=None):
    """The points under a sweep root: the finished ones (sweep.json) in a table with the best, the unfinished or
    failed ones listed and skipped; works on a half done sweep."""
    root = Path(root)
    finished = []
    skipped = []
    for child in sorted(path for path in root.iterdir() if path.is_dir()):
        info = child / "sweep.json"
        if info.exists():
            point = json.loads(info.read_text())
            history = read_history(child)
            finished.append({**point, "dir": str(child), "turns": len(history),
                             "last": _last_values(history, ("val/", "test/"))})
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
    objective = finished[0]["objective"]
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
    import pandas

    pandas.DataFrame(rows).to_csv(target / "sweep.csv", index=False)
    text = root_markdown(result)
    (target / "sweep.md").write_text(text)
    return "sweep", text, str(target)


def root_markdown(result):
    rows = result["points"]
    columns = ["id", *[key for key in rows[0] if key not in ("id", "objective", "turn", "turns") and "/" not in key],
               "objective", "turn", "turns", *sorted({key for row in rows for key in row if "/" in key})]
    objective = result["objective"]
    lines = [f"# sweep: {objective['monitor']} ({objective['mode']}, {objective['at']})", "",
             "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column)
            cells.append(f"{value:.6g}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    best = result["best"]
    lines += ["", f"best: point {best['id']} {best['point']} with {objective['monitor']}={best['value']:.6g} at turn "
                  f"{best['turn']} ({best['dir']})"]
    if result["skipped"]:
        lines.append("skipped: " + ", ".join(f"{entry['dir']} ({entry['status']})" for entry in result["skipped"]))
    return "\n".join(lines) + "\n"


def collect(run_dirs, out=None):
    """Summarize runs: a sweep root (one directory of points) gives sweep.csv, sweep.json, sweep.md and the best
    point; run directories give cv.json and cv.md when every run has a fold param, sweep.json and sweep.md
    otherwise."""
    if len(run_dirs) == 1 and Path(run_dirs[0]).is_dir() and not is_run_dir(run_dirs[0]):
        return collect_root(run_dirs[0], out)
    runs = load_runs(run_dirs)
    if not runs:
        raise ValueError("no run with a resolved.yaml among the given directories")
    target = Path(out) if out is not None else Path(runs[0]["dir"]).parent
    target.mkdir(parents=True, exist_ok=True)
    if all((entry["config"].get("params") or {}).get("fold") is not None for entry in runs):
        result = fold_summary(runs)
        (target / "cv.json").write_text(json.dumps(result, indent=2, default=float))
        text = fold_markdown(result)
        (target / "cv.md").write_text(text)
        return "cv", text, str(target)
    result = sweep_table(runs)
    (target / "sweep.json").write_text(json.dumps(result, indent=2, default=float))
    text = sweep_markdown(result)
    (target / "sweep.md").write_text(text)
    return "sweep", text, str(target)
