import json
from pathlib import Path

from kalfa.registration import lego
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


@lego("/lego/kalfa/data_report", returns="data_report", bus=["record"],
      description="The shape of the data at every stage of the data block, read from the bus keys the stages "
                  "wrote: the rows and columns of the source and after every transform, the sets after the "
                  "split and after their transforms, the fitted frame transforms and preprocessors, the features "
                  "and targets, the loaders; written to the record as data.json")
def data_report(stages, split, after, fitted, prep, frames, loaders, record=None):
    report = {
        "stages": stage_entries(stages),
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
        (Path(record) / "data.json").write_text(json.dumps(report, indent=2))
    logger.info(f"{report['fit']['features']} features, {len(report['fit']['targets'])} targets; "
                f"{', '.join(f'{name} {entry['rows']}' for name, entry in report['sets'].items())} rows")
    return report
