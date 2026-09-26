import ast
import concurrent.futures
import difflib
import json
import math
import os
import re
import shutil
import statistics
import textwrap
from pathlib import Path

import pandas

from . import collect_figures
from .describe.text import PLAIN, field_line, head, table, width_of
from .record import Record, failure_text, read_resolved
from .std.common.history import History, is_number


NUMERIC = re.compile(r"^-?(\d[\d.eE+\-/]*|nan|inf)$")
REPORTS = "reports"
IMAGES = (".png", ".svg", ".jpg", ".jpeg", ".gif", ".webp")


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


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(item).replace("|", "\\|").replace("\n", " ") for item in row) + " |"
              for row in rows]
    return lines


def joined(names):
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def scored(line):
    return {key: value for key, value in (line or {}).items() if key.startswith(("val/", "test/")) and is_number(value)}


def line_at(history, turn):
    return next((line for line in history if line.get("turn") == turn), history[-1] if len(history) else {})


def load_runs(run_dirs):
    runs = []
    for directory in run_dirs:
        directory = Path(directory)
        if not (directory / "resolved.yaml").exists():
            continue
        runs.append({"dir": str(directory), "config": read_resolved(directory), "history": History.read(directory)})
    return runs


def default_out(run_dirs):
    first = Path(run_dirs[0])
    return (first if len(run_dirs) == 1 else first.parent) / REPORTS


def checkpoint_monitor(config):
    checkpoint = ((config or {}).get("training") or {}).get("checkpoint")
    params = (checkpoint.get("params") or {}) if isinstance(checkpoint, dict) else {}
    return params.get("monitor"), params.get("mode") or "min"


def reported_line(entry):
    config = entry["config"] or {}
    monitor, mode = checkpoint_monitor(config)
    if (config.get("training") or {}).get("report") != "best" or not monitor:
        return None
    try:
        _, turn = History(entry["history"]).best(monitor, mode)
    except ValueError:
        return None
    return line_at(entry["history"], turn)


def fold_summary(runs):
    folds = []
    for entry in runs:
        params = entry["config"].get("params") or {}
        reported = reported_line(entry)
        folds.append({"dir": entry["dir"], "fold": params.get("fold"), "turns": len(entry["history"]),
                      "state": Record(entry["dir"]).state(), "last": History(entry["history"]).last(),
                      **({"reported": {"turn": reported.get("turn"), **scored(reported)}} if reported else {})})
    folds.sort(key=lambda item: (item["fold"] is None, item["fold"]))
    keys = sorted({key for fold in folds for key in fold["last"]})
    summary = {}
    for key in keys:
        mean, std = mean_std([fold["last"].get(key, math.nan) for fold in folds])
        summary[key] = {"mean": mean, "std": std}
    result = {"folds": folds, "summary": summary}
    reported = [fold["reported"] for fold in folds if "reported" in fold]
    if reported:
        names = sorted({key for line in reported for key in line if key != "turn"})
        result["reported"] = {key: dict(zip(("mean", "std"), mean_std([line.get(key, math.nan) for line in reported])))
                              for key in names}
    return result


def fold_markdown(result, extras=None):
    extras = extras or {}
    keys = list(result["summary"])
    lines = ["# k fold summary", "", "| metric | mean | std |", "|---|---|---|"]
    for key in keys:
        entry = result["summary"][key]
        lines.append(f"| {key} | {cell(entry['mean'])} | {cell(entry['std'])} |")
    lines += ["", "| fold | turns | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
    for fold in result["folds"]:
        values = " | ".join(cell(fold["last"].get(key, math.nan)) for key in keys)
        lines.append(f"| {fold['fold']} | {fold['turns']} | {values} |")
    lines += ["", "The tables above take every fold at its last turn."]
    if result.get("reported"):
        names = list(result["reported"])
        lines += ["", "## At the reported turn", "",
                  "training.report is best: every fold at the best turn of its checkpoint monitor, the weights its "
                  "predictions and plots come from.", ""]
        lines += md_table(["metric", "mean", "std"], [[key, cell(value["mean"]), cell(value["std"])]
                                                      for key, value in result["reported"].items()])
        lines += [""] + md_table(["fold", "turn", *names], [
            [fold["fold"], fold["reported"]["turn"], *[cell(fold["reported"].get(key, math.nan)) for key in names]]
            for fold in result["folds"] if "reported" in fold])
    lines += ["", "## Folds", ""]
    lines += md_table(["fold", "state", "turns", "record"], [
        [fold["fold"], fold.get("state", ""), fold["turns"], link(fold["dir"], extras.get("target"))]
        for fold in result["folds"]])
    unhealthy = [fold for fold in result["folds"] if fold.get("state") not in (None, "finished")]
    if unhealthy:
        lines += ["", f"{len(unhealthy)} fold(s) did not finish; the means above include what they reached."]
    for path, text in extras.get("figures", []):
        lines += ["", "## Curves", "", f"![{text}]({path})"]
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


def point_status(child):
    state = Record(child).state()
    return state if state in ("failed", "lost") else "unfinished"


def finite(value):
    return is_number(value) and math.isfinite(value)


def ranked_points(finished, objective):
    sign = 1 if objective.get("mode", "min") == "min" else -1
    return sorted(finished, key=lambda entry: (not finite(entry["objective"]["value"]),
                                               sign * entry["objective"]["value"]
                                               if finite(entry["objective"]["value"]) else 0))


def link(directory, target):
    text = Path(directory).name
    if target is None:
        return text
    return f"[{text}]({os.path.relpath(Path(directory), Path(target))})"


def listed(declared):
    if isinstance(declared, list):
        return declared
    found = re.search(r"Choices\(values=(\[[^\]]*\])", declared if isinstance(declared, str) else "")
    if found:
        try:
            return list(ast.literal_eval(found.group(1)))
        except (ValueError, SyntaxError):
            pass
    return None


def param_axis(declared, values):
    levels = listed(declared)
    if levels is not None:
        return levels
    text = declared if isinstance(declared, str) else ""
    present = [value for value in values if value is not None]
    if present and all(finite(value) for value in present):
        return "log" if "log=True" in text else "linear"
    try:
        return sorted(set(present))
    except TypeError:
        return sorted(set(present), key=str)


def copy_plots(source, destination):
    images = sorted(path for path in (Path(source) / "plots").glob("*") if path.suffix.lower() in IMAGES)
    if not images:
        return []
    destination.mkdir(parents=True, exist_ok=True)
    for path in images:
        shutil.copy2(path, destination / path.name)
    return [path.name for path in images]


def config_diff(first, second, limit=120):
    texts = []
    for entry in (first, second):
        path = Path(entry["dir"]) / "resolved.yaml"
        texts.append(path.read_text().splitlines() if path.exists() else [])
    lines = list(difflib.unified_diff(texts[0], texts[1], f"point {first['id']}", f"point {second['id']}", n=1,
                                      lineterm=""))
    return lines[:limit] + ([f"... and {len(lines) - limit} more lines"] if len(lines) > limit else [])


def sweep_figures(target, finished, highlighted, objective, space, unit, grouped=None):
    from .std.common.figure import Figure

    figures, written = Figure(), []
    monitor = objective["monitor"]
    over = grouped["over"] if grouped else None
    entries = [{**entry, "label": f"{over} {cell(entry['point'].get(over))}" if over else f"point {entry['id']}"}
               for entry in finished]
    shown = [next(item for item in entries if item["dir"] == entry["dir"]) for entry in highlighted]
    path = collect_figures.curves(figures, target, entries, shown, monitor, unit)
    if path is not None:
        written.append((path, f"{monitor} over the {unit}s" + (", the best setting in colour" if over and shown
                                                                else "")))
    rated = setting_entries(grouped) if grouped else finished
    label = f"mean {monitor} over {over}" if over else monitor
    for key in list(rated[0]["point"]) if rated else []:
        values = [entry["point"].get(key) for entry in rated]
        axis = param_axis(space.get(key), values)
        scores = [entry["objective"]["value"] for entry in rated]
        path = collect_figures.against(figures, target, key, values, scores, label, log=axis == "log",
                                       levels=axis if isinstance(axis, list) else None,
                                       noun="settings" if over else "points")
        if path is not None:
            written.append((path, f"{label} against {key}"))
    return [(os.path.relpath(path, target), text) for path, text in written]


def settings_over(root, finished, over, space, objective):
    swept = list(finished[0]["point"])
    if over not in swept:
        raise ValueError(f"{root}: {over} is no swept param; the swept params are {joined(swept)}")
    levels = listed(space.get(over))
    if levels is None:
        raise ValueError(f"{root}: the space gives {over} no list of values; --mean-over averages over the values "
                         f"the space lists")
    groups = {}
    for entry in finished:
        setting = {key: value for key, value in entry["point"].items() if key != over}
        groups.setdefault(json.dumps(setting, sort_keys=True, default=str), (setting, []))[1].append(entry)
    settings = []
    for setting, entries in groups.values():
        entries.sort(key=lambda entry: levels.index(entry["point"][over]) if entry["point"][over] in levels
                     else len(levels))
        kept = [entry for entry in entries if finite(entry["objective"]["value"])]
        scores = [entry["objective"]["value"] for entry in kept]
        reached = [entry["point"][over] for entry in kept]
        mean, std = mean_std(scores) if scores else (math.nan, math.nan)
        settings.append({
            "setting": setting, "mean": mean, "std": std, "min": min(scores) if scores else math.nan,
            "max": max(scores) if scores else math.nan, "count": len(scores),
            "missing": [level for level in levels if level not in reached],
            "points": [{"id": entry["id"], "level": entry["point"][over], "value": entry["objective"]["value"],
                        "turn": entry["objective"]["turn"], "dir": entry["dir"]} for entry in entries],
            "metrics": {key: dict(zip(("mean", "std"), mean_std([entry["at"].get(key, math.nan) for entry in kept])))
                        for key in metrics_of([entry["at"] for entry in kept])}})
    sign = 1 if objective.get("mode", "min") == "min" else -1
    return {"over": over, "levels": levels, "keys": [key for key in swept if key != over],
            "ranked": sorted((item for item in settings if not item["missing"]), key=lambda item: sign * item["mean"]),
            "incomplete": [item for item in settings if item["missing"]]}


def setting_entries(grouped):
    return [{"point": item["setting"], "objective": {"value": item["mean"]}} for item in grouped["ranked"]]


def setting_text(setting):
    return ", ".join(f"{name}={cell(value)}" for name, value in setting.items())


def settings_frame(grouped):
    every = grouped["ranked"] + grouped["incomplete"]
    names = sorted({name for item in every for name in item["metrics"]})
    columns = [*grouped["keys"], "mean", "std", "min", "max", "values", "missing",
               *[f"{name} {part}" for name in names for part in ("mean", "std")]]
    rows = [{**item["setting"], "mean": item["mean"], "std": item["std"], "min": item["min"], "max": item["max"],
             "values": item["count"], "missing": " ".join(cell(level) for level in item["missing"]),
             **{f"{name} {part}": item["metrics"].get(name, {}).get(part) for name in names
                for part in ("mean", "std")}} for item in every]
    return pandas.DataFrame(rows, columns=columns)


def paired_diff(grouped):
    if len(grouped["ranked"]) < 2:
        return [], None
    first, second = grouped["ranked"][:2]
    for point in first["points"]:
        other = next((item for item in second["points"] if item["level"] == point["level"]), None)
        if other is not None:
            return config_diff(point, other), point["level"]
    return [], None


def settings_markdown(grouped, objective, target, top):
    over, keys, ranked = grouped["over"], grouped["keys"], grouped["ranked"]
    monitor = objective["monitor"]
    lines = ["", f"## Best setting, the mean over {over}", ""]
    if not ranked:
        lines.append(f"No setting has a finite objective for every value of {over} yet.")
    else:
        best = ranked[0]
        lines += [f"{setting_text(best['setting']) or 'The only setting'} reaches a mean {monitor} of "
                  f"{cell(best['mean'])} with a standard deviation of {cell(best['std'])} over the {best['count']} "
                  f"values of {over}.", ""]
        lines += md_table([over, "point", "objective", "turn"], [
            [cell(point["level"]), link(point["dir"], target), cell(point["value"]), point["turn"]]
            for point in best["points"]])
        if best["metrics"]:
            lines += [""] + md_table(["metric at the objective turn", "mean", "std"], [
                [name, cell(value["mean"]), cell(value["std"])] for name, value in best["metrics"].items()])
        shown = ranked[:top]
        lines += ["", f"## Top {len(shown)} setting{'s' if len(shown) != 1 else ''}", ""]
        lines += md_table(["rank", *keys, "mean", "std", "values", "gap to the best"], [
            [position + 1, *[cell(item["setting"].get(key)) for key in keys], cell(item["mean"]), cell(item["std"]),
             item["count"], cell(abs(item["mean"] - best["mean"]))] for position, item in enumerate(shown)])
    if grouped["incomplete"]:
        lines += ["", f"## Settings without every value of {over}", "",
                  f"A setting is ranked once every value of {over} has a finished point with a finite objective.", ""]
        lines += md_table([*keys, "finished", "missing"], [
            [*[cell(item["setting"].get(key)) for key in keys],
             ", ".join(cell(point["level"]) for point in item["points"] if finite(point["value"])),
             ", ".join(cell(level) for level in item["missing"])] for item in grouped["incomplete"]])
    return lines


def settings_text(grouped, objective, style, width):
    over, keys, ranked = grouped["over"], grouped["keys"], grouped["ranked"]
    total = len(ranked) + len(grouped["incomplete"])
    lines = block(f"MEAN OVER {over.upper()}", width, style)
    lines += wrapped("settings", f"{len(ranked)} of {total} with every value of {over}", style, width)
    if ranked:
        body = [[*[cell(item["setting"].get(key)) for key in keys], cell(item["mean"]), cell(item["std"]),
                 str(item["count"])] for item in ranked]
        lines += ["", *numbered([*keys, "mean", "std", "values"], body, style, width), ""]
        best = ranked[0]
        lines += wrapped("best", f"{setting_text(best['setting']) or 'the only setting'}, mean "
                                 f"{objective['monitor']} = {cell(best['mean'])}, std {cell(best['std'])}", style,
                         width)
    return lines


def spearman(first, second):
    value = pandas.Series(first).rank().corr(pandas.Series(second).rank())
    return float(value) if finite(value) else None


def importance(rated, keys, space, objective):
    better = -1 if objective.get("mode", "min") == "min" else 1
    pick = min if better < 0 else max
    rows = []
    for key in keys:
        pairs = [(entry["point"].get(key), entry["objective"]["value"]) for entry in rated
                 if finite(entry["objective"]["value"]) and entry["point"].get(key) is not None]
        axis = param_axis(space.get(key), [value for value, _ in pairs])
        if isinstance(axis, list):
            groups = {}
            for value, score in pairs:
                groups.setdefault(json.dumps(value, sort_keys=True, default=str), (value, []))[1].append(score)
            if len(groups) < 2 or len(pairs) <= len(groups):
                continue
            scores = [score for _, score in pairs]
            mean = statistics.fmean(scores)
            spread = sum((score - mean) ** 2 for score in scores)
            if spread <= 0:
                continue
            between = sum(len(items) * (statistics.fmean(items) - mean) ** 2 for _, items in groups.values())
            level = pick(groups.values(), key=lambda item: statistics.fmean(item[1]))[0]
            rows.append({"param": key, "measure": "η²", "value": between / spread,
                         "direction": f"best mean at {cell(level)}", "count": len(pairs)})
            continue
        numeric = [(float(value), score) for value, score in pairs if finite(value)]
        rho = spearman(*zip(*numeric)) if len(numeric) >= 3 else None
        if rho is None:
            continue
        direction = "" if rho == 0 else "better as it grows" if rho * better > 0 else "better as it shrinks"
        rows.append({"param": key, "measure": "Spearman ρ", "value": rho, "direction": direction,
                     "count": len(numeric)})
    return rows


def level_rows(finished, key, levels, objective):
    pick = min if objective.get("mode", "min") == "min" else max
    rows = []
    for level in levels:
        scores = [entry["objective"]["value"] for entry in finished
                  if entry["point"].get(key) == level and finite(entry["objective"]["value"])]
        if scores:
            rows.append([cell(level), len(scores), cell(pick(scores)), cell(statistics.median(scores))])
        else:
            rows.append([cell(level), 0, "", ""])
    return rows


def counts_line(result):
    rows, skipped = result["points"], result["skipped"]
    planned = result.get("planned") or len(rows) + len(skipped)
    parts = [f"{len(rows)} of {planned} points finished"]
    for status in ("failed", "lost", "unfinished"):
        count = sum(1 for entry in skipped if entry["status"] == status)
        if count:
            parts.append(f"{count} {status}")
    started = len(rows) + len(skipped)
    if isinstance(result.get("planned"), int) and result["planned"] > started:
        parts.append(f"{result['planned'] - started} not started")
    return ", ".join(parts) + "."


def root_markdown(result, extras=None):
    extras = extras or {}
    rows, objective, best = result["points"], result["objective"], result["best"]
    target = extras.get("target")
    monitor = objective["monitor"]
    swept = swept_of(rows[0])
    by_id = {row["id"]: row for row in rows}
    lines = [f"# sweep: {monitor} ({objective['mode']}, {objective['at']})", "", counts_line(result)]
    if result.get("strategy"):
        lines += ["", f"Strategy {result['strategy']}."]
    if extras.get("grouped"):
        lines += settings_markdown(extras["grouped"], objective, target, extras.get("top", 5))
    lines += ["", "## Best point", "",
              f"Point {best['id']} ({link(best['dir'], target)}) reaches {monitor} = {cell(best['value'])} at turn "
              f"{best['turn']} of {by_id[best['id']]['turns']}.", ""]
    lines += md_table(["param", "value"], [[name, cell(value)] for name, value in best["point"].items()])
    metrics = {key: value for key, value in by_id[best["id"]].items() if "/" in key}
    if metrics:
        lines += [""] + md_table([f"metric at turn {best['turn']}", "value"],
                                 [[key, cell(value)] for key, value in sorted(metrics.items())])
    for name in extras.get("copies", []):
        lines += ["", f"![{name}](best/{name})"]
    ranked = extras.get("ranked") or []
    top = ranked[:extras.get("top", 5)]
    if top:
        first = top[0]["objective"]["value"]
        lines += ["", f"## Top {len(top)}", ""]
        lines += md_table(["rank", "id", *swept, "objective", "gap to the best", "turn", "turns"], [
            [position + 1, entry["id"], *[cell(entry["point"].get(key)) for key in swept],
             cell(entry["objective"]["value"]),
             cell(abs(entry["objective"]["value"] - first)) if finite(entry["objective"]["value"]) else "",
             entry["objective"]["turn"], entry["turns"]] for position, entry in enumerate(top)])
    drawn = extras.get("figures", [])
    param_figures = [(path, text) for path, text in drawn if Path(path).name.startswith("param_")]
    levels = extras.get("levels", {})
    weights = result.get("importance") or []
    if param_figures or levels or weights:
        lines += ["", "## Parameters"]
        if weights:
            lines += [""] + md_table(["param", "measure", "value", "direction", extras.get("scored", "points")], [
                [row["param"], row["measure"], cell(row["value"]), row["direction"], row["count"]] for row in weights])
            lines += ["", "ρ is the Spearman rank correlation of a numeric param with the objective, η² the share of "
                          "the objective's variance the levels of a choice explain. Both look at one param at a time: "
                          "on points not drawn from a grid, a param chosen together with another can look more or "
                          "less important than it is."]
        for path, text in param_figures:
            lines += ["", f"![{text}]({path})"]
        for key, table_rows in levels.items():
            lines += [""] + md_table([key, extras.get("scored", "points"), "best", "median"], table_rows)
    for path, text in drawn:
        if Path(path).name.startswith("curves"):
            lines += ["", "## Curves", "", f"![{text}]({path})"]
    if result["skipped"]:
        lines += ["", "## Failed and unfinished", ""]
        lines += md_table(["point", "status", "reason"], [[entry["dir"], entry["status"], entry.get("reason") or ""]
                                                         for entry in result["skipped"]])
    if extras.get("diff"):
        lines += ["", f"## {extras.get('diff_title') or 'Config, the best against the runner-up'}", "", "```diff",
                  *extras["diff"], "```"]
    columns = ["id", *swept, "objective", "turn", "turns", *metrics_of(rows)]
    lines += ["", f"<details><summary>Every finished point ({len(rows)})</summary>", ""]
    lines += md_table(columns, [[cell(row.get(column)) for column in columns] for row in rows])
    lines += ["", "</details>"]
    return "\n".join(lines) + "\n"


def read_point(child):
    if not (is_run_dir(child) or (child / "run.json").exists()):
        return None
    info = child / "sweep.json"
    if not info.exists():
        status = point_status(child)
        return "skipped", {"dir": child.name, "status": status,
                           "reason": failure_text(Record(child)) if status == "failed" else None}
    try:
        point = json.loads(info.read_text())
        turn = point["objective"]["turn"]
        if "value" not in point["objective"] or "id" not in point or "point" not in point:
            raise KeyError("objective value, id or point")
    except (ValueError, KeyError, TypeError):
        return "skipped", {"dir": child.name, "status": "unfinished",
                           "reason": "sweep.json is being written or is broken"}
    history = History.read(child)
    return "finished", {**point, "dir": str(child), "turns": len(history), "history": history,
                        "at": scored(line_at(history, turn))}


def collect_root(root, out=None, markdown=False, style=PLAIN, top=5, figures=True, mean_over=None):
    root = Path(root)
    manifest = Record(root).read_json("manifest.json") or {}
    finished = []
    skipped = []
    children = sorted(path for path in root.iterdir() if path.is_dir())
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for found in pool.map(read_point, children):
            if found is not None:
                (finished if found[0] == "finished" else skipped).append(found[1])
    if not finished:
        raise ValueError(f"{root}: no finished point (a directory with sweep.json) under the sweep root")
    objective = {**finished[0]["objective"], **(manifest.get("objective") or {})}
    ranked = ranked_points(finished, objective)
    best = ranked[0]
    rows = []
    for entry in finished:
        row = {"id": entry["id"], **entry["point"], "objective": entry["objective"]["value"],
               "turn": entry["objective"]["turn"], "turns": entry["turns"], **entry["at"]}
        rows.append(row)
    result = {"objective": {key: objective.get(key) for key in ("monitor", "mode", "at")}, "points": rows,
              "best": {"id": best["id"], "point": best["point"], "value": best["objective"]["value"],
                       "turn": best["objective"]["turn"], "dir": best["dir"]},
              "skipped": skipped, "planned": manifest.get("total"), "strategy": manifest.get("strategy")}
    space = manifest.get("space") or {}
    grouped = settings_over(root, finished, mean_over, space, objective) if mean_over is not None else None
    if grouped:
        result["groups"] = grouped
    rated = setting_entries(grouped) if grouped else finished
    keys = (grouped["keys"] if grouped else list(best["point"])) if rated else []
    result["importance"] = importance(rated, keys, space, objective)
    target = Path(out) if out is not None else root / REPORTS
    target.mkdir(parents=True, exist_ok=True)
    (target / "sweep.json").write_text(json.dumps(result, indent=2, default=float))
    pandas.DataFrame(rows).to_csv(target / "sweep.csv", index=False)
    if grouped:
        settings_frame(grouped).to_csv(target / "groups.csv", index=False)
    unit = "epoch" if (Record(best["dir"]).read_json("manifest.json") or {}).get("turn") == "epoch" else "turn"
    levels = {}
    for key in keys:
        axis = param_axis(space.get(key), [entry["point"].get(key) for entry in rated])
        if isinstance(axis, list):
            levels[key] = level_rows(rated, key, axis, objective)
    if grouped:
        dirs = {point["dir"] for point in grouped["ranked"][0]["points"]} if grouped["ranked"] else set()
        highlighted = [entry for entry in finished if entry["dir"] in dirs]
        diff, level = paired_diff(grouped)
        diff_title = f"Config, the best setting against the runner-up at {grouped['over']} = {cell(level)}"
    else:
        highlighted = ranked[:top]
        diff = config_diff(ranked[0], ranked[1]) if len(ranked) > 1 else []
        diff_title = None
    extras = {"target": target, "ranked": ranked, "top": top, "levels": levels, "grouped": grouped,
              "scored": "settings" if grouped else "points", "diff": diff, "diff_title": diff_title,
              "copies": copy_plots(best["dir"], target / "best"),
              "figures": sweep_figures(target, finished, highlighted, objective, space, unit, grouped)
              if figures else []}
    report = root_markdown(result, extras)
    (target / "sweep.md").write_text(report)
    written = ["sweep.csv", "sweep.json", "sweep.md", *(["groups.csv"] if grouped else []),
               *[path for path, _ in extras["figures"]], *[f"best/{name}" for name in extras["copies"]]]
    return "sweep", report if markdown else root_text(result, style), str(target), written


def skipped_by_status(result):
    grouped = []
    for status in ("failed", "lost", "unfinished"):
        names = [entry["dir"] for entry in result["skipped"] if entry["status"] == status]
        if names:
            grouped.append((status, names))
    return grouped


def best_point(best):
    return ", ".join(f"{name}={cell(value)}" for name, value in best["point"].items())


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
    if result.get("groups"):
        lines += ["", *settings_text(result["groups"], objective, style, width)]
    lines.append("")
    return "\n".join(lines) + "\n"


def fold_figures(target, runs, result):
    from .std.common.figure import Figure

    monitor = next((checkpoint_monitor(entry["config"])[0] for entry in runs if checkpoint_monitor(entry["config"])[0]),
                   None)
    if monitor is None:
        monitor = next((key for entry in runs for line in entry["history"] for key in line if key.startswith("val/")),
                       None)
    if monitor is None:
        return []
    folds = {fold["dir"]: fold["fold"] for fold in result["folds"]}
    entries = [{**entry, "label": f"fold {folds.get(entry['dir'])}"} for entry in runs]
    unit = "epoch" if (Record(runs[0]["dir"]).read_json("manifest.json") or {}).get("turn") == "epoch" else "turn"
    path = collect_figures.curves(Figure(), target, entries, entries, monitor, unit)
    return [(os.path.relpath(path, target), f"{monitor} over the {unit}s, one line per fold")] if path else []


def collect(run_dirs, out=None, markdown=False, style=PLAIN, top=5, figures=True, mean_over=None):
    if len(run_dirs) == 1 and Path(run_dirs[0]).is_dir() and not is_run_dir(run_dirs[0]):
        return collect_root(run_dirs[0], out, markdown, style, top, figures, mean_over)
    if mean_over is not None:
        raise ValueError("--mean-over takes one sweep root (a directory whose manifest says sweep)")
    runs = load_runs(run_dirs)
    if not runs:
        raise ValueError("no run with a resolved.yaml among the given directories")
    target = Path(out) if out is not None else default_out(run_dirs)
    target.mkdir(parents=True, exist_ok=True)
    if all((entry["config"].get("params") or {}).get("fold") is not None for entry in runs):
        result = fold_summary(runs)
        (target / "cv.json").write_text(json.dumps(result, indent=2, default=float))
        drawn = fold_figures(target, runs, result) if figures else []
        report = fold_markdown(result, {"target": target, "figures": drawn})
        (target / "cv.md").write_text(report)
        written = ["cv.json", "cv.md", *[path for path, _ in drawn]]
        return "cv", report if markdown else fold_text(result, style), str(target), written
    result = sweep_table(runs)
    (target / "sweep.json").write_text(json.dumps(result, indent=2, default=float))
    pandas.DataFrame(result["rows"], columns=sweep_columns(result)).to_csv(target / "sweep.csv", index=False)
    report = sweep_markdown(result)
    (target / "sweep.md").write_text(report)
    return "sweep", report if markdown else sweep_text(result, style), str(target), ["sweep.csv", "sweep.json",
                                                                                     "sweep.md"]
