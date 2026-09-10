from ..std.pre.base import assign_fields
from .text import short


def data_params(prepared):
    document = prepared.document
    if document is None:
        return None
    return ((document.get("flow") or {}).get("data") or {}).get("params") or {}


def block_params(prepared, name):
    document = prepared.document
    if document is None:
        return {}
    return ((document.get("flow") or {}).get(name) or {}).get("params") or {}


def field_plan(prepared):
    data = prepared.surface.data.get("data") if isinstance(prepared.surface.data, dict) else None
    data = data if isinstance(data, dict) else {}
    fields = data.get("fields") if isinstance(data.get("fields"), dict) else {}
    drop = data.get("drop") if isinstance(data.get("drop"), list) else []
    names = list(data.get("preprocessors")) if isinstance(data.get("preprocessors"), dict) else []
    return fields, drop, names


def batch_size(params):
    loaders = params.get("loaders") or {}
    loader = loaders.get("train") or next(iter(loaders.values()), None) or {}
    return (loader.get("params") or {}).get("size")


def owners_of(prepared):
    header = prepared.header
    if header is None:
        return {}, []
    fields, drop, _ = field_plan(prepared)
    columns = [name for name in header["columns"] if name not in drop]
    owners, _ = assign_fields(columns, list(fields))
    return owners, columns


def target_fields(prepared, probe=None):
    if probe is not None and probe.prep is not None:
        return [item.name for item in probe.prep.fields if item.target]
    fields, _, _ = field_plan(prepared)
    owners, columns = owners_of(prepared)
    found = []
    for pattern, spec in fields.items():
        if (spec or {}).get("target"):
            found.extend([column for column in columns if owners.get(column) == pattern])
    return found


def unwrap(call):
    if isinstance(call, dict) and short(call.get("uri")) in ("criterion", "metric", "objective"):
        params = call.get("params") or {}
        inner = params.get("criterion") or params.get("metric") or params.get("objective")
        if isinstance(inner, dict):
            return inner
    return call


def group_of(document, name):
    return (document or {}).get(name) or {}


def reference(text):
    return text.split(".", 1)[-1] if isinstance(text, str) and text.startswith("@") else text
