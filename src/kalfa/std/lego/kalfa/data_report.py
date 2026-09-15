from pathlib import Path

from kalfa.std.common.files import write_json
from kalfa.std.common.log import logger_for
from kalfa.std.pre.base import Grouped, columns_of_source


logger = logger_for("data.report")


def rows_of(value):
    try:
        return int(len(value))
    except TypeError:
        return None


def by_set(table, suffix):
    return {name.removesuffix(suffix): table[name] for name in sorted(table or {})}


def stage_entries(stages):
    entries = []
    previous = None
    for key in sorted(stages or {}, key=lambda key: int(key.rsplit("_", 1)[-1])):
        columns = columns_of_source(stages[key])
        entry = {"stage": key, "rows": rows_of(stages[key]), "columns": len(columns)}
        if previous is not None:
            entry["added"] = [column for column in columns if column not in previous]
            entry["removed"] = [column for column in previous if column not in columns]
        entries.append(entry)
        previous = columns
    return entries


def fitted_counts(prep):
    counts = {}
    for name, entry in prep.fitted.items():
        counts[name] = len(entry.columns) if isinstance(entry, Grouped) else len(entry)
    return counts


def calls_of(items):
    return [{"uri": item.get("lego"), "params": item.get("with") or {}} for item in items or []]


def data_report(stages, split, after, fitted, prep, frames, loaders, transforms=None, set_transforms=None,
                record=None):
    entries = stage_entries(stages)
    for position, call in enumerate(calls_of(transforms)):
        if position + 1 < len(entries):
            entries[position + 1]["call"] = call
    report = {
        "stages": entries,
        "set_transforms": {name: calls_of(items) for name, items in (set_transforms or {}).items() if items},
        "split": {name: rows_of(value) for name, value in by_set(split, "_df_0").items()},
        "after_set_transforms": {name: rows_of(value) for name, value in by_set(after, "_df_1").items()},
        "frames": [type(item).__name__ for item in fitted or []],
        "fit": {"fields": len(prep.fields), "features": len(prep.features), "targets": list(prep.targets),
                "preprocessors": fitted_counts(prep),
                "extras": [extra for item in prep.fields for extra in item.extras]},
        "sets": {name: {"rows": rows_of(frame), "features": len(frame.features)}
                 for name, frame in by_set(frames, "_frame").items()},
        "loaders": {name: {"batches": rows_of(loader), "size": loader.batch_size}
                    for name, loader in by_set(loaders, "_loader").items()},
    }
    if record is not None:
        write_json(Path(record) / "data.json", report)
    logger.info(f"{report['fit']['features']} features, {len(report['fit']['targets'])} targets; "
                f"{', '.join(f'{name} {entry['rows']}' for name, entry in report['sets'].items())} rows")
    return report
