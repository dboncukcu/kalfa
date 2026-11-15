from pathlib import Path

from .build import fixed_of, mutable_paths, node_signature
from .context import train_of
from .errors import RunError
from .record import version as kalfa_version
from .refs import jsonable, ref_json, ref_parts


FORMAT = 1
SET_PREFIX = {"train": "train", "valid": "val", "test": "test"}
DEFAULT_ON = ("train", "valid", "test")


def plan_of(tree, version=None):
    settings = tree.settings or {}
    nodes = {}
    for path, node in tree.nodes.items():
        signature = node_signature(node.target, node.slot) if node.target is not None else None
        checked = signature is not None and not signature.accepts_extra
        nodes[path] = {
            "slot": node.slot,
            "use": node.use,
            "params": jsonable(node.params),
            "keys": jsonable(node.keys),
            "fixed": list(fixed_of(node)),
            "signature": "checked" if checked else "open",
        }
    return {
        "format": FORMAT,
        "kalfa": version if version is not None else kalfa_version(),
        "seed": jsonable(settings.get("seed")),
        "device": jsonable(settings.get("device")),
        "record": jsonable(settings.get("record")),
        "nodes": nodes,
        "data": data_of(tree),
        "train": jsonable(settings.get("train") or {}),
        "rules": rules_of(tree),
        "series": series_of(tree),
        "mutable": mutable_paths(tree),
    }


def data_of(tree):
    node = tree.nodes.get("data")
    if node is None:
        return {"inputs": [], "exposes": []}
    target = node.target
    inputs = getattr(target, "inputs", None) if target is not None else None
    exposes = getattr(target, "exposes", None) if target is not None else None
    out = {"inputs": list(inputs) if inputs else ["x"], "exposes": list(exposes or ())}
    fields = fields_of(tree, node)
    if fields is not None:
        out["fields"] = fields
    return out


def fields_of(tree, node):
    file = node.params.get("file")
    if not isinstance(file, str) or not file:
        return None
    candidates = [Path(file)]
    for directory in getattr(tree.config, "directories", None) or ():
        candidates.append(Path(directory) / file)
    for candidate in candidates:
        if candidate.is_file():
            try:
                return read_header(candidate)
            except Exception:
                return None
    return None


def read_header(path):
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        import pyarrow.parquet as parquet

        schema = parquet.read_schema(str(path))
    elif suffix in (".csv", ".tsv"):
        import pyarrow.csv as csv

        options = csv.ParseOptions(delimiter="\t") if suffix == ".tsv" else None
        reader = csv.open_csv(str(path), parse_options=options)
        try:
            schema = reader.schema
        finally:
            close = getattr(reader, "close", None)
            if callable(close):
                close()
    else:
        return None
    return {name: str(schema.field(name).type) for name in schema.names}


def call_of(tree, path, value):
    if isinstance(value, str):
        return {"use": value}
    node = tree.nodes.get(path)
    if node is None or not isinstance(value, dict) or "use" not in value:
        return jsonable(value)
    return {"use": value["use"], **jsonable(node.keys), **jsonable(node.params)}


def patch_value(value):
    if ref_parts(value) is None:
        return jsonable(value)
    try:
        return ref_json(value)
    except RunError:
        return jsonable(value)


def rules_of(tree):
    out = []
    for spec in tree.rules:
        out.append({
            "id": spec.name,
            "when": call_of(tree, f"rules.{spec.name}.when", spec.when),
            "after": spec.after,
            "once": bool(spec.once),
            "patch": [{"path": str(path), "value": patch_value(value)} for path, value in spec.set.items()],
            "stop": bool(spec.stop),
            "run": call_of(tree, f"rules.{spec.name}.run", spec.run),
        })
    return out


def sets_of(node):
    on = node.keys.get("on")
    if on is None:
        on = getattr(node.target, "on", None) if node.target is not None else None
    if on is None:
        on = DEFAULT_ON
    if isinstance(on, str):
        on = (on,)
    return [SET_PREFIX.get(str(name), str(name)) for name in on]


def series_of(tree):
    series = []
    for node in tree.nodes.values():
        if node.slot == "loss":
            series.append(f"train/{node.name}")
            series.extend(f"{prefix}/{node.name}" for prefix in sets_of(node))
        elif node.slot == "metric":
            series.extend(f"{prefix}/{node.name}" for prefix in sets_of(node))
    for node in tree.nodes.values():
        if node.slot == "optimizer":
            series.extend((f"lr/{node.name}", f"minimizes/{node.name}"))
    track = train_of(tree).get("track")
    if isinstance(track, str):
        track = [track]
    for path in track or ():
        series.append(f"track/{path}")
    return list(dict.fromkeys(series))
