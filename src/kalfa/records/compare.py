import difflib
from pathlib import Path

from .reader import series_of, summary_of


RESOLVED = "resolved.yaml"


def differing_params(summaries):
    params = [dict((summary or {}).get("params") or {}) for summary in summaries]
    keys = []
    for item in params:
        for key in item:
            if key not in keys:
                keys.append(key)
    shared = {}
    differing = {}
    for key in keys:
        values = [item.get(key) for item in params]
        everywhere = all(key in item for item in params)
        if everywhere and all(value == values[0] for value in values[1:]):
            shared[key] = values[0]
        else:
            differing[key] = values
    return {"shared": shared, "differing": differing}


def aligned_series(directories, keys):
    if isinstance(keys, str):
        keys = [keys]
    names = [str(directory) for directory in directories]
    tables = {}
    for name in names:
        table = series_of(name, keys)
        tables[name] = table if isinstance(table, dict) else {}
        if keys is None:
            keys = [key for key in tables[name] if key != "turn"]
    keys = list(keys or [])
    turns = set()
    for table in tables.values():
        turns.update(turn for turn in table.get("turn") or [] if turn is not None)
    turns = sorted(turns)
    aligned = {"turn": turns}
    for name in names:
        table = tables[name]
        positions = {turn: index for index, turn in enumerate(table.get("turn") or []) if turn is not None}
        aligned[name] = {}
        for key in keys:
            values = table.get(key) or []
            aligned[name][key] = [values[positions[turn]] if turn in positions and positions[turn] < len(values)
                                  else None for turn in turns]
    return aligned


def config_of(directory):
    path = Path(directory) / RESOLVED
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def config_diff(first, second):
    texts = []
    for directory in (first, second):
        text = config_of(directory)
        if text is None:
            return None
        texts.append(text.splitlines())
    lines = difflib.unified_diff(texts[0], texts[1], fromfile=Path(first).name, tofile=Path(second).name,
                                 lineterm="")
    return "\n".join(lines)


def compare(directories, keys=None):
    directories = list(directories)
    records = [summary_of(directory) for directory in directories]
    names = [Path(directory).name for directory in directories]
    diffs = {}
    for directory, name in zip(directories[1:], names[1:]):
        diffs[f"{names[0]} vs {name}"] = config_diff(directories[0], directory)
    return {"records": records, "params": differing_params(records), "series": aligned_series(directories, keys),
            "diffs": diffs}
