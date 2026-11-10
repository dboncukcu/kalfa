import numbers
from pathlib import Path

from matplotlib.ticker import MaxNLocator

from kalfa import summary as summarizing
from kalfa.commands.report.page import (
    color_of,
    embedded,
    escape,
    finite,
    grid,
    new_figure,
    paragraph,
    plural,
    shown,
    table,
)
from kalfa.commands.report.predictions import predictions_part
from kalfa.core.errors import KalfaError
from kalfa.core.record import now, version
from kalfa.records import reader
from kalfa.records import sweep as sweeping
from kalfa.std.plots import LEGEND_LIMIT, dress, spans_decades


CURVES = 8
DETAILS = 20
LISTED = 6
KNOWN = 12
CURVE_PREFIXES = ("val/", "train/")


def sources(summary):
    if summary.sweep is not None:
        for record in summary.records:
            if isinstance(record.get("sweep"), dict):
                return [str(Path(record["path"]).parent)]
    return [str(record["path"]) for record in summary.records]


def title_of(summary):
    if summary.sweep is not None and summary.records:
        return f"kalfa report: {Path(sources(summary)[0]).name}"
    if len(summary.records) == 1:
        return f"kalfa report: {summary.records[0].get('name')}"
    return f"kalfa report: {plural(len(summary.records), 'record')}"


def heading(summary, title, sections):
    paths = sources(summary)
    listed = ", ".join(paths[:LISTED])
    if len(paths) > LISTED:
        listed += f" and {len(paths) - LISTED} more"
    links = "".join(f'<a href="#{key}">{escape(name)}</a>' for key, name in sections)
    return (f"<header><h1>{escape(title)}</h1>"
            f"{paragraph(f'written {now()} by kalfa {version()}', 'muted')}"
            f"{paragraph(listed, 'muted')}<nav>{links}</nav></header>")


def ranked(summary):
    pool = list(summary.records)
    names = summary.table["name"].tolist() if "name" in summary.table.columns else []
    ordered = []
    for name in names:
        for position, record in enumerate(pool):
            if record.get("name") == name:
                ordered.append(pool.pop(position))
                break
    return ordered


def best_line(summary):
    if summary.by is None:
        return "no series to rank by"
    best = summary.best
    if best is None:
        return f"no record has a finite {summary.by} yet"
    turn = f" at turn {shown(best.get('best_turn'))}" if best.get("best_turn") is not None else ""
    return f"best: {best.get('name')}, {summary.by} {shown(best.get('best'))}{turn} ({summary.mode})"


def summary_section(summary):
    columns = summarizing.columns_of(summary, "all")
    rows = list(summary.table[columns].itertuples(index=False, name=None)) if columns else []
    parts = ['<h2 id="summary">Summary</h2>']
    if summary.by is not None:
        parts.append(paragraph(f"{plural(len(rows), 'record')} ranked by {summary.by} ({summary.mode})", "muted"))
    parts.append(table(columns, rows) if rows else paragraph("no records", "note"))
    parts.append(paragraph(best_line(summary), "best"))
    return "\n".join(parts)


def folds_section(summary):
    parts = ['<h2 id="folds">Folds</h2>']
    for group in summary.folds or []:
        names = summarizing.fold_series(summary, group)
        count = len(group.get("members") or [])
        parts.append(paragraph(f"{summarizing.fold_label(group)}: mean and std over {plural(count, 'record')}",
                               "muted"))
        parts.append(table(["series", "mean", "std"], [(name, group["mean"][name], group["std"].get(name))
                                                       for name in names]))
    return "\n".join(parts)


def history_tables(records):
    return [(str(record.get("name")), reader.series_of(record["path"])) for record in records]


def curve_names(tables, by, curves):
    if curves is False:
        return [], 0
    if isinstance(curves, str):
        curves = [part.strip() for part in curves.split(",") if part.strip()]
    if curves is not None:
        return list(dict.fromkeys(str(name) for name in curves)), 0
    present = []
    for _, found in tables:
        for key, values in found.items():
            if key in present or not str(key).startswith(CURVE_PREFIXES):
                continue
            if any(finite(value) for value in values):
                present.append(key)
    present.sort(key=lambda key: (key != by, not key.startswith("val/"), key))
    return present[:CURVES], max(len(present) - CURVES, 0)


def curve_figure(name, tables, word):
    figure, axis = new_figure()
    drawn, values = 0, []
    for position, (label, found) in enumerate(tables):
        pairs = [(turn, value) for turn, value in zip(found.get("turn") or [], found.get(name) or [])
                 if turn is not None and finite(value)]
        if not pairs:
            continue
        turns, ys = zip(*pairs)
        axis.plot(turns, ys, label=label, color=color_of(position), linewidth=1.6,
                  marker="o" if len(pairs) < 3 else None, markersize=3)
        drawn += 1
        values.extend(ys)
    if not drawn:
        figure.clear()
        return None
    if spans_decades(values):
        axis.set_yscale("log")
    if drawn <= LEGEND_LIMIT:
        axis.legend(frameon=False, fontsize="small")
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.set_title(name, loc="left")
    axis.set_xlabel(word)
    axis.set_ylabel(name)
    dress(axis)
    return embedded(figure, f"{name} over the {word}s of {plural(drawn, 'record')}")


def curves_section(summary, records, curves):
    tables = history_tables(records)
    names, extra = curve_names(tables, summary.by, curves)
    word = "epoch" if records and all(record.get("unit") == "epochs" for record in records) else "turn"
    figures = [figure for figure in (curve_figure(name, tables, word) for name in names) if figure]
    parts = ['<h2 id="curves">Curves</h2>']
    if figures:
        parts.append(grid(figures))
    elif names and curves is not None:
        parts.append(paragraph(f"no finite values for {', '.join(names)} in any history", "note"))
    else:
        parts.append(paragraph("no train/ or val/ series in any history yet", "note"))
    if extra:
        parts.append(paragraph(f"{extra} more series not drawn; the report draws at most {CURVES}", "muted"))
    return "\n".join(parts)


def point_label(point):
    identity = point.get("id") if isinstance(point, dict) else None
    if isinstance(identity, numbers.Integral) and not isinstance(identity, bool):
        return f"{int(identity):04d}"
    return str(identity)


def is_log(spec, positions):
    if isinstance(spec, dict) and "log" in spec:
        return all(position > 0 for position in positions)
    return all(position > 0 for position in positions) and max(positions) / min(positions) >= 10.0


def param_figure(name, points, watch, spec, best):
    pairs = [((point.get("values") or {}).get(name), point["objective"], point is best) for point in points
             if sweeping.is_valued(point) and name in (point.get("values") or {})]
    if not pairs:
        return None
    figure, axis = new_figure(5.6, 3.6)
    values = [value for value, _, _ in pairs]
    if all(finite(value) for value in values):
        positions = [float(value) for value in values]
        if is_log(spec, positions):
            axis.set_xscale("log")
        dress(axis)
    else:
        labels = [shown(value) for value in spec] if isinstance(spec, list) else []
        for value in values:
            if shown(value) not in labels:
                labels.append(shown(value))
        positions = [labels.index(shown(value)) for value in values]
        dress(axis)
        axis.set_xticks(range(len(labels)), labels, rotation=30 if len(labels) > 6 else 0)
    objectives = [objective for _, objective, _ in pairs]
    axis.scatter(positions, objectives, s=30, color=color_of(0), alpha=0.8, linewidths=0, label="points")
    chosen = [(position, objective) for position, (_, objective, top) in zip(positions, pairs) if top]
    if chosen:
        axis.scatter([chosen[0][0]], [chosen[0][1]], s=110, facecolors="none", edgecolors=color_of(1),
                     linewidths=1.8, label="best")
        axis.legend(frameon=False, fontsize="small")
    axis.set_title(f"{watch} against {name}", loc="left")
    axis.set_xlabel(name)
    axis.set_ylabel(watch)
    return embedded(figure, f"{watch} against {name} over {plural(len(pairs), 'point')} with an objective")


def varied(name, points, spec):
    seen = {shown((point.get("values") or {}).get(name)) for point in points if name in (point.get("values") or {})}
    if isinstance(spec, dict):
        return True
    if isinstance(spec, list):
        seen.update(shown(value) for value in spec)
    return len(seen) > 1


def sweep_section(summary):
    sweep = summary.sweep if isinstance(summary.sweep, dict) else {}
    objective = sweep.get("objective") if isinstance(sweep.get("objective"), dict) else {}
    space = sweep.get("space") if isinstance(sweep.get("space"), dict) else {}
    points = sorted(summary.points, key=sweeping.id_key)
    names = [str(name) for name in space]
    for point in points:
        for name in point.get("values") or {}:
            if name not in names:
                names.append(name)
    watch = str(objective.get("watch") or "objective")
    strategy = sweep.get("strategy")
    strategy = strategy.get("use") if isinstance(strategy, dict) else strategy
    states = {}
    for point in points:
        states[str(point.get("status"))] = states.get(str(point.get("status")), 0) + 1
    planned = sweep.get("points")
    counted = plural(len(points), "point")
    if isinstance(planned, numbers.Integral) and not isinstance(planned, bool):
        counted += f" of {planned} planned"
    facts = [f"strategy {strategy or 'unknown'}",
             f"objective {watch} ({sweeping.mode_of(objective)}, at {objective.get('at') or 'best'})", counted,
             *(f"{count} {state}" for state, count in sorted(states.items()))]
    best = sweeping.best_point(points, objective)
    parts = ['<h2 id="sweep">Sweep</h2>', paragraph(", ".join(facts), "muted")]
    if best is not None:
        parts.append(paragraph(f"best point {point_label(best)}: {watch} {shown(best.get('objective'))}", "best"))
    else:
        parts.append(paragraph(f"no point has a finite {watch} yet", "best"))
    rows = [[point_label(point), *((point.get("values") or {}).get(name) for name in names), point.get("status"),
             point.get("turn"), point.get("objective")] for point in points]
    parts.append(table(["id", *names, "status", "turn", watch], rows))
    figures = [figure for figure in (param_figure(name, points, watch, space.get(name), best) for name in names
                                     if varied(name, points, space.get(name))) if figure]
    if figures:
        parts.append(grid(figures))
    return "\n".join(parts)


def normal_key(key):
    key = str(key).strip()
    return str(int(key)) if key.isdigit() else key


def wanted_of(points):
    if points is None:
        return None
    if isinstance(points, str):
        if points.strip().lower() == "all":
            return "all"
        points = points.split(",")
    elif not isinstance(points, (list, tuple, set)):
        points = [points]
    keys = [normal_key(item) for item in points if str(item).strip()]
    return list(dict.fromkeys(keys)) or None


def record_keys(record):
    keys = {normal_key(record.get("name")), normal_key(Path(record["path"]).name)}
    sweep = record.get("sweep")
    if isinstance(sweep, dict) and sweep.get("id") is not None:
        keys.add(normal_key(sweep["id"]))
    return keys


def chosen_records(summary, records, points):
    wanted = wanted_of(points)
    if wanted is None:
        return records[:DETAILS], max(len(records) - DETAILS, 0)
    if wanted == "all":
        return list(records), 0
    chosen, seen, missing = [], set(), []
    for key in wanted:
        found = [record for record in summary.records if key in record_keys(record)]
        if not found:
            missing.append(key)
        for record in found:
            if id(record) not in seen:
                seen.add(id(record))
                chosen.append(record)
    if missing:
        known = [str(record.get("name")) for record in summary.records[:KNOWN]]
        more = " and more" if len(summary.records) > KNOWN else ""
        raise KalfaError(f"no record or point {', '.join(missing)} in the report; it holds {', '.join(known)}{more}")
    return chosen, 0


def best_text(best):
    if not isinstance(best, dict) or not best.get("series"):
        return "no checkpoint watch to name a best"
    if best.get("value") is None:
        return f"best {best['series']}: no finite value yet"
    return f"best {best['series']} {shown(best['value'])} at turn {shown(best.get('turn'))} ({best.get('mode')})"


def record_section(record, stats):
    path = str(record["path"])
    total = f" of {record['total']}" if record.get("total") is not None else ""
    facts = (f"{record.get('status')}, turn {record.get('turn')}{total}, "
             f"{shown(record.get('seconds'), 'seconds') or 0} s, {path}")
    parts = [f'<section class="record"><h3>{escape(record.get("name"))}</h3>', paragraph(facts, "muted")]
    if record.get("failure"):
        parts.append(paragraph(f"failed: {record['failure']}", "note"))
    parts.append(paragraph(best_text(record.get("best")), "best"))
    params = record.get("params") or {}
    last = record.get("last") or {}
    blocks = []
    if params:
        blocks.append("<div><h4>params</h4>" + table(["param", "value"], list(params.items())) + "</div>")
    if last:
        rows = sorted(last.items(), key=lambda item: summarizing.series_key(item[0]))
        blocks.append("<div><h4>last values</h4>" + table(["series", "last"], rows) + "</div>")
    if blocks:
        parts.append('<div class="pair">' + "".join(blocks) + "</div>")
    if stats:
        parts.append(predictions_part(path))
    parts.append("</section>")
    return "\n".join(part for part in parts if part)


def details_section(chosen, rest, stats):
    parts = ['<h2 id="records">Records</h2>']
    if rest:
        parts.append(paragraph(f"the first {len(chosen)} of {len(chosen) + rest} records in the ranking; "
                               f"--points IDS names the others", "muted"))
    if not chosen:
        parts.append(paragraph("no records", "note"))
    parts.extend(record_section(record, stats) for record in chosen)
    return "\n".join(parts)
