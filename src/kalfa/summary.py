import math
import numbers
import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas

from .core.errors import KalfaError
from .records import reader
from .records import sweep as sweeping


FIXED = ("name", "status", "turn", "total", "seconds")
RESERVED = (*FIXED, "best", "best_turn")
HIDDEN = ("train/", "lr/")
SETS = {"train": 0, "val": 1, "test": 2}
normal_mode = reader.normal_mode


@dataclass
class Summary:
    records: list
    table: object
    by: str | None
    mode: str
    best: dict | None = None
    folds: list | None = None
    sweep: dict | None = None
    points: list = field(default_factory=list)

    def to_markdown(self):
        return markdown_of(self)


def under(root):
    root = Path(root)
    found = []
    for entry in reader.records_under(root):
        if entry.get("kind") == "sweep":
            found.extend(Path(point) for point in sweeping.point_dirs(root / entry["path"]))
        else:
            found.append(root / entry["path"])
    return found


def expand(paths):
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    directories, seen, root = [], set(), None
    for given in paths:
        path = Path(given)
        if reader.is_record(path):
            found = [path]
        elif sweeping.is_sweep(path):
            found = [Path(point) for point in sweeping.point_dirs(path)]
            if not found:
                raise KalfaError(f"no records under {given}")
            if root is None:
                root = path
        elif path.is_dir():
            found = under(path)
            if not found:
                raise KalfaError(f"no records under {given}")
        else:
            raise KalfaError(f"not a record, a sweep root or a directory of records: {given}")
        for directory in found:
            key = directory.resolve()
            if key not in seen:
                seen.add(key)
                directories.append(str(directory))
    sweep = sweeping.read_sweep(root) if root is not None else None
    points = sweeping.sweep_points(root) if root is not None else []
    return directories, sweep, points


def default_series(summaries):
    if not summaries:
        return None, "min"
    best = summaries[0].get("best")
    if isinstance(best, dict) and best.get("series"):
        return str(best["series"]), normal_mode(best.get("mode"))
    last = summaries[0].get("last") or {}
    by = next((key for key in last if str(key).startswith("val/")), None)
    return by, "min"


def value_of(summary, by, mode):
    if by is None or summary.get("status") == "failed":
        return None, None
    best = summary.get("best")
    if isinstance(best, dict) and best.get("series") == by and normal_mode(best.get("mode")) == normal_mode(mode):
        return best.get("value"), best.get("turn")
    return reader.best_in(reader.read_history(summary["path"]), by, normal_mode(mode))


def sort_key(row, maximize):
    value = row["best"]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return (1, 0.0)
    return (0, -float(value) if maximize else float(value))


def series_key(series):
    name = str(series)
    return (SETS.get(name.split("/", 1)[0], len(SETS)), name)


def column_of(param):
    key = str(param)
    if key in RESERVED or key.startswith("last_"):
        return f"param_{key}"
    return key


def table_of(summaries, by, mode):
    params, series = [], []
    for summary in summaries:
        for key in summary.get("params") or {}:
            if key not in params:
                params.append(key)
        for key in summary.get("last") or {}:
            if key != by and key not in series:
                series.append(key)
    columns = [*FIXED, *(column_of(key) for key in params), "best", "best_turn"]
    if by is not None:
        columns.append(f"last_{by}")
    series.sort(key=series_key)
    columns.extend(f"last_{key}" for key in series)
    rows = []
    for summary in summaries:
        given = summary.get("params") or {}
        last = summary.get("last") or {}
        row = {key: summary.get(key) for key in FIXED}
        row.update({column_of(key): given.get(key) for key in params})
        row["best"], row["best_turn"] = value_of(summary, by, mode)
        if by is not None:
            row[f"last_{by}"] = last.get(by)
        row.update({f"last_{key}": last.get(key) for key in series})
        rows.append(row)
    maximize = normal_mode(mode) == "max"
    rows.sort(key=lambda row: sort_key(row, maximize))
    return pandas.DataFrame(rows, columns=columns)


def clean(value):
    if isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isnan(value):
        return None
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return float(value)
    return value


def summarize(paths, by=None, mode=None, top=None):
    directories, sweep, points = expand(paths)
    summaries = [reader.summary_of(directory) for directory in directories]
    default_by, default_mode = default_series(summaries)
    by = by if by is not None else default_by
    mode = normal_mode(mode if mode is not None else default_mode)
    table = table_of(summaries, by, mode)
    if top is not None:
        table = table.head(int(top)).reset_index(drop=True)
    best = {key: clean(value) for key, value in table.iloc[0].to_dict().items()} if len(table) else None
    if best is not None and best["best"] is None:
        best = None
    folds = None
    if any("fold" in (summary.get("params") or {}) for summary in summaries):
        folds = sweeping.group_folds(summaries)
    return Summary(records=summaries, table=table, by=by, mode=mode, best=best, folds=folds, sweep=sweep,
                   points=list(points or []))


def cell(value, column=None):
    value = clean(value)
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format(value, ".6g" if column == "seconds" else ".4g")
    return str(value)


def hidden(column, by):
    name = str(column)
    if not name.startswith("last_"):
        return False
    series = name[len("last_"):]
    return series != by and series.startswith(HIDDEN)


def columns_of(summary, columns):
    present = [str(column) for column in summary.table.columns]
    if columns is None:
        return [column for column in present if not hidden(column, summary.by)]
    if isinstance(columns, str):
        if columns == "all":
            return present
        columns = [part.strip() for part in columns.split(",") if part.strip()]
    chosen = [str(column) for column in columns]
    unknown = [column for column in chosen if column not in present]
    if unknown:
        raise KalfaError(f"unknown columns: {', '.join(unknown)}")
    return chosen


def rows_of(summary, top, columns=None):
    table = summary.table
    if top is not None:
        table = table.head(int(top))
    chosen = columns_of(summary, columns)
    table = table[chosen]
    return chosen, [[cell(value, column) for value, column in zip(row, chosen)]
                    for row in table.itertuples(index=False, name=None)]


def text_of(summary, top=None, columns=None):
    columns, rows = rows_of(summary, top, columns)
    widths = [max([len(column), *(len(row[position]) for row in rows)]) for position, column in enumerate(columns)]
    lines = ["  ".join(column.ljust(width) for column, width in zip(columns, widths)).rstrip()]
    for row in rows:
        lines.append("  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip())
    if not rows:
        lines.append("no records")
    return "\n".join(lines)


def fold_label(group):
    members = group.get("members") or []
    folds = ", ".join(cell((member.get("params") or {}).get("fold")) for member in members)
    where = ", ".join(f"{key}={cell(value)}" for key, value in (group.get("group") or {}).items())
    return f"folds {folds} of {where}" if where else f"folds {folds}"


def fold_series(summary, group, every=False):
    names = sorted(group.get("mean") or {}, key=series_key)
    if every:
        return names
    return [name for name in names if name == summary.by or not name.startswith(HIDDEN)]


def folds_table(summary):
    params = []
    for group in summary.folds or []:
        for key in group.get("group") or {}:
            if key not in params:
                params.append(key)
    rows = []
    for group in summary.folds or []:
        given = group.get("group") or {}
        for name in fold_series(summary, group, every=True):
            rows.append([*(given.get(key) for key in params), len(group.get("members") or []), name,
                         group["mean"][name], group["std"].get(name)])
    return pandas.DataFrame(rows, columns=[*(column_of(key) for key in params), "records", "series", "mean", "std"])


def folds_text(summary):
    lines = []
    for group in summary.folds or []:
        names = fold_series(summary, group)
        width = max([len("series"), *(len(name) for name in names)])
        lines.extend([f"{fold_label(group)}: mean and std over {len(group.get('members') or [])} records",
                      f"  {'series'.ljust(width)}  {'mean'.ljust(10)}  std"])
        lines.extend(f"  {name.ljust(width)}  {cell(group['mean'][name]).ljust(10)}  {cell(group['std'].get(name))}"
                     for name in names)
    return "\n".join(lines)


def escape(text):
    return text.replace("|", "\\|")


def markdown_of(summary, top=None, columns="all"):
    columns, rows = rows_of(summary, top, columns)
    lines = ["| " + " | ".join(escape(column) for column in columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join(escape(value) for value in row) + " |")
    return "\n".join(lines)


def write(summary, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "summary.csv"
    markdown = out / "summary.md"
    summary.table.to_csv(csv, index=False, float_format="%.12g")
    markdown.write_text(markdown_of(summary) + "\n", encoding="utf-8")
    if not summary.folds:
        return str(csv), str(markdown)
    folds = out / "folds.csv"
    folds_table(summary).to_csv(folds, index=False, float_format="%.12g")
    return str(csv), str(markdown), str(folds)
