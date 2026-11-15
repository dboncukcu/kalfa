import numpy
import pandas

from .tables import finite, label_text, labels, numbers_of, numeric, ordered_labels, texts


CODES = (3, 20)


def plan_nodes(plan):
    nodes = (plan or {}).get("nodes")
    return nodes if isinstance(nodes, dict) else {}


def plan_targets(plan):
    node = plan_nodes(plan).get("data")
    params = node.get("params") if isinstance(node, dict) else None
    targets = params.get("targets") if isinstance(params, dict) else None
    if isinstance(targets, dict) and targets:
        return [str(name) for name in targets]
    if isinstance(targets, (list, tuple)) and targets:
        return [str(name) for name in targets]
    if isinstance(targets, str):
        return [targets]
    fields = ((plan or {}).get("data") or {}).get("fields")
    if isinstance(fields, dict):
        return [str(name) for name, field in fields.items() if isinstance(field, dict) and field.get("target")]
    return []


def plan_classes(plan):
    return labels(((plan or {}).get("data") or {}).get("classes"))


def fitted_classes(fields, target=None):
    if not isinstance(fields, dict):
        return None
    if target is not None:
        for item in fields.get("target_fields") or ():
            if isinstance(item, dict) and item.get("name") == target:
                return labels(item.get("classes"))
    return labels(fields.get("classes"))


def fitted_entries(fields):
    if not isinstance(fields, dict):
        return {}
    return {str(item["name"]): labels(item.get("classes")) for item in fields.get("target_fields") or ()
            if isinstance(item, dict) and item.get("name") is not None}


def classes_for(plan, fitted, target=None):
    entries = fitted_entries(fitted)
    if target in entries and (entries[target] is None or sum(1 for found in entries.values() if found) > 1):
        return entries[target]
    return plan_classes(plan) or fitted_classes(fitted, target)


def plan_pairings(plan):
    pairs = {}
    for path, node in plan_nodes(plan).items():
        if not isinstance(node, dict) or node.get("slot") not in ("loss", "metric"):
            continue
        keys = node.get("keys") if isinstance(node.get("keys"), dict) else {}
        pred, target = keys.get("pred"), keys.get("target")
        if isinstance(pred, str) and isinstance(target, str):
            pairs.setdefault(pred, target)
    return pairs


def frame_targets(frame):
    return [column for column in frame.columns
            if column != "row" and not str(column).startswith(("pred_", "raw_", "prob_"))]


def kind_of(frame, target, classes, scores):
    if classes or scores:
        return "classification"
    return "regression" if pandas.api.types.is_numeric_dtype(frame[target]) else "classification"


def spread_member(column, columns):
    stem, _, position = str(column).rpartition("_")
    if stem and position.isdigit() and f"{stem}_0" in columns and f"{stem}_1" in columns:
        return stem, int(position)
    return None


def declared_targets(plan, fitted, frame):
    targets = plan_targets(plan)
    if targets:
        return targets
    if isinstance(fitted, dict) and isinstance(fitted.get("targets"), list):
        return [str(name) for name in fitted["targets"]]
    found = frame_targets(frame)
    return found if len(found) == 1 else []


def matches(frame, plan, fitted=None):
    columns = [str(column) for column in frame.columns]
    pairings = plan_pairings(plan)
    targets = declared_targets(plan, fitted, frame)
    fallback = targets[0] if len(targets) == 1 else None
    found = []
    for column in columns:
        if not column.startswith("pred_"):
            continue
        output = column[len("pred_"):]
        target, base = pairings.get(output), None
        member = None if target is not None else spread_member(column, columns)
        if member is not None:
            base = pairings.get(member[0][len("pred_"):], fallback)
            target = None if base is None else f"{base}_{member[1]}"
        elif target is None:
            target = fallback
        if target is None or target not in columns or target == column:
            target = None
        found.append((column, output, target, base))
    return found


def raw_of(columns, output):
    return f"raw_{output}" if f"raw_{output}" in columns else None


def scores_of(columns, output):
    return [name for name in columns if name.startswith(f"prob_{output}_")]


def suffix_label(suffix, known):
    if suffix in known:
        return suffix
    try:
        number = float(suffix)
    except ValueError:
        return suffix
    return label_text(number) or suffix


def score_labels(scores, output, known=()):
    prefix = f"prob_{output}_"
    known = set(known or ())
    return {column: suffix_label(column[len(prefix):], known) for column in scores if column.startswith(prefix)}


def label_columns(scores, output, known=()):
    found = {}
    for column, label in score_labels(scores, output, known).items():
        found.setdefault(label, column)
    return found


def prediction_pairs(frame, plan, fitted=None):
    if frame is None:
        return []
    columns = [str(column) for column in frame.columns]
    pairs = []
    for column, output, target, base in matches(frame, plan, fitted):
        if target is None:
            continue
        raw, scores = raw_of(columns, output), scores_of(columns, output)
        kind = kind_of(frame, target, classes_for(plan, fitted, base or target), scores)
        pairs.append({"output": output, "pred": column, "target": target, "kind": kind, "raw": raw, "scores": scores})
    return pairs


def observed(frame, pred, target):
    return ordered_labels(set(texts(frame[target]).tolist()) | set(texts(frame[pred]).tolist()))


def detected(frame, pred, target, classes, scores):
    truth = frame[target]
    if classes or scores or not numeric(truth):
        found = classes or observed(frame, pred, target)
        if len(found) == 2:
            return "binary", found, ["binary", "multiclass"]
        return "multiclass", found, ["multiclass"]
    distinct = numpy.unique(finite(numbers_of(truth)))
    if len(distinct) == 2:
        return "binary", [label_text(value) for value in distinct], ["binary", "regression"]
    low, high = CODES
    if numeric(frame[pred]) and low <= len(distinct) <= high and bool(numpy.all(distinct == numpy.round(distinct))):
        guessed = numpy.unique(finite(numbers_of(frame[pred])))
        if bool(numpy.isin(guessed, distinct).all()):
            return "multiclass", [label_text(value) for value in distinct], ["multiclass", "regression"]
    return "regression", None, ["regression"]


def target_pair(frame, column, output, target, classes):
    columns = [str(name) for name in frame.columns]
    scores = scores_of(columns, output)
    auto, found, tasks = detected(frame, column, target, classes, scores)
    return {"output": output, "pred": column, "columns": [column], "target": target, "auto": auto, "tasks": tasks,
            "raw": raw_of(columns, output), "scores": scores, "classes": found}


def loose_pair(output, pred, task, columns, raw=None):
    return {"output": output, "pred": pred, "columns": columns, "target": None, "auto": task, "tasks": [task],
            "raw": raw, "scores": [], "classes": None}


def task_pairs(frame, plan, fitted=None):
    columns = [str(column) for column in frame.columns]
    found, spreads = [], {}
    for column, output, target, base in matches(frame, plan, fitted):
        if target is not None:
            found.append(target_pair(frame, column, output, target, classes_for(plan, fitted, base or target)))
            continue
        member = spread_member(column, columns)
        if member is None:
            found.append(loose_pair(output, column, "values", [column], raw_of(columns, output)))
            continue
        stem = member[0]
        if stem not in spreads:
            spreads[stem] = loose_pair(stem[len("pred_"):], stem, "embedding", [])
            found.append(spreads[stem])
        spreads[stem]["columns"].append(column)
    return [single(pair, columns) if pair["auto"] == "embedding" and len(pair["columns"]) < 2 else pair
            for pair in found]


def single(pair, columns):
    column = pair["columns"][0]
    output = column[len("pred_"):]
    return loose_pair(output, column, "values", [column], raw_of(columns, output))
