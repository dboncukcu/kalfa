import math
from pathlib import Path

from kalfa.core import aliases
from kalfa.std.data import splits


LISTED = 6
CLIP = 90
SCALARS = (bool, int, float, str)
LABELS = ("name", "path")


def alias_of(target, slot=None):
    kind = target if isinstance(target, type) else type(target)
    path = aliases.target_path(kind)
    found = [name for name, entry in aliases.ALIASES.items() if aliases.target_path(entry.target) == path]
    preferred = [name for name in found if slot is None or aliases.ALIASES[name].slot == slot]
    chosen = preferred or found
    return chosen[0] if chosen else None


def kind_of(data):
    return alias_of(data, "data") or aliases.target_path(type(data))


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item) for item in value]
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if callable(getattr(value, "tolist", None)):
        return jsonable(value.tolist())
    return str(value)


def number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def amount(value):
    value = number(value)
    if value is None:
        return "?"
    if abs(value) < 10000:
        return str(value)
    if abs(value) < 1000000:
        return f"{value / 1000:.1f}k".replace(".0k", "k")
    return f"{value / 1000000:.1f}M".replace(".0M", "M")


def clip(text, size=CLIP):
    text = "" if text is None else str(text)
    return text if len(text) <= size else text[:size - 1].rstrip() + "…"


def tail(text, size=60):
    text = str(text)
    return text if len(text) <= size else "…" + text[-(size - 1):]


def listed(names, limit=LISTED):
    names = [str(name) for name in names or ()]
    if len(names) <= limit:
        return ", ".join(names)
    return ", ".join(names[:limit]) + f" and {len(names) - limit} more"


def share(value):
    return f"{round(float(value) * 100, 1):g}%"


def simple(value):
    if isinstance(value, SCALARS):
        return True
    return isinstance(value, (list, tuple)) and len(value) <= 8 and all(isinstance(item, SCALARS) for item in value)


def value_text(value):
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(value_text(item) for item in value) + "]"
    return str(value)


def params_text(params):
    return ", ".join(f"{key} {value_text(value)}" for key, value in params.items() if simple(value))


def params_of(thing):
    fields = getattr(thing, "__dict__", None)
    if not isinstance(fields, dict):
        return ""
    return params_text({key: value for key, value in fields.items()
                        if key not in LABELS and not key.startswith("_") and not key.endswith("_")})


def named(text):
    text = str(text)
    if aliases.is_import_path(text):
        return text.rsplit(".", 1)[-1], text
    return text, ""


def step_label(item):
    if isinstance(item, str):
        return named(item)
    if isinstance(item, dict):
        label, path = named(item.get("use"))
        return label, params_text({key: value for key, value in item.items() if key != "use"}) or path
    return alias_of(item, "value") or type(item).__name__, params_of(item)


def steps_of(spec, chain=()):
    items = [] if spec is None else list(spec) if isinstance(spec, (list, tuple)) else [spec]
    if chain and len(items) != len(chain):
        items = list(chain)
    return [step_label(item) for item in items]


def pipeline_steps(pipeline):
    names = list(getattr(pipeline, "names", None) or ())
    steps = list(getattr(pipeline, "steps", None) or ())
    if len(names) != len(steps):
        return [step_label(step) for step in steps]
    found = []
    for name, step in zip(names, steps):
        label, path = named(name)
        found.append((label, params_of(step) or path))
    return found


class Flow:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.groups = []
        self.ids = set()

    def unique(self, key):
        found, position = str(key), 1
        while found in self.ids:
            position += 1
            found = f"{key}#{position}"
        self.ids.add(found)
        return found

    def node(self, key, kind, label, detail="", count=None, group=None):
        key = self.unique(key)
        self.nodes.append({"id": key, "kind": kind, "label": str(label), "detail": clip(detail), "count": number(count),
                           "group": group})
        return key

    def edge(self, start, end, label=""):
        if start is not None and end is not None and start != end:
            self.edges.append({"from": start, "to": end, "label": str(label or "")})

    def join(self, starts, end, label=""):
        for start in starts:
            self.edge(start, end, label)

    def group(self, key, label, detail="", parent=None):
        key = self.unique(key)
        self.groups.append({"id": key, "label": str(label), "detail": clip(detail), "parent": parent})
        return key

    def step(self, start, key, kind, label, detail="", count=None, group=None, edge=""):
        made = self.node(key, kind, label, detail, count, group)
        self.edge(start, made, edge)
        return made

    def chain(self, starts, key, steps, kind="chain", label=None, detail=""):
        steps = list(steps)
        if not steps:
            return list(starts)
        group = self.group(f"group:{key}", label or key, detail) if label is not None else None
        ends = list(starts)
        for position, (name, about) in enumerate(steps):
            made = self.node(f"{key}:{position}", kind, name, about, group=group)
            self.join(ends, made)
            ends = [made]
        return ends

    def result(self):
        return {"nodes": self.nodes, "edges": self.edges, "groups": self.groups}


def source_node(graph, path, count=None, extra=()):
    text = str(path)
    name = Path(text).name or text
    about = ([tail(text)] if text != name else []) + [part for part in extra if part]
    return graph.node("source", "source", name, ", ".join(about), count)


def split_label(split):
    name = alias_of(split, "value")
    return f"{name} split" if name else type(split).__name__


def split_detail(split):
    if isinstance(split, splits.KFoldSplit):
        parts = [f"fold {split.fold} of {split.folds}"]
        if split.valid_ratio:
            parts.append(f"valid {share(split.valid_ratio)} of the rest")
        if split.seed is not None:
            parts.append(f"seed {split.seed}")
        return ", ".join(parts)
    if isinstance(split, splits.GivenSplit):
        return f"by the column {split.column}"
    shares = ", ".join(share(value) for value in getattr(split, "ratios", None) or () if value)
    if isinstance(split, splits.SequentialSplit):
        about = "in file order" + (f", within each {split.group}" if split.group is not None else "")
    elif isinstance(split, splits.RandomSplit):
        about = "shuffled" + (f" with seed {split.seed}" if split.seed is not None else "")
    else:
        return params_of(split)
    return f"{shares}, {about}" if shares else about


def shares_of(split):
    if isinstance(split, splits.KFoldSplit):
        found = {"train": "other folds", "test": f"fold {split.fold}"}
        if split.valid_ratio:
            found["valid"] = f"{share(split.valid_ratio)} of the rest"
        return found
    if isinstance(split, splits.GivenSplit):
        labels = {"train": split.train, "valid": split.valid, "test": split.test}
        return {name: f"{split.column} = {value}" for name, value in labels.items() if value is not None}
    ratios = getattr(split, "ratios", None)
    if isinstance(ratios, (list, tuple)) and len(ratios) == len(splits.SET_NAMES):
        try:
            return {name: share(value) for name, value in zip(splits.SET_NAMES, ratios) if float(value) > 0}
        except (TypeError, ValueError):
            return {}
    return {}


def split_node(graph, start, split):
    return graph.step(start, "split", "split", split_label(split), split_detail(split))


def set_nodes(graph, start, counts, details=None, edges=None):
    details = details or {}
    edges = edges or {}
    return {name: graph.step(start, f"set:{name}", "set", name, details.get(name, ""), count, edge=edges.get(name, ""))
            for name, count in (counts or {}).items()}


def batch_text(batch):
    if not isinstance(batch, dict):
        return ""
    parts = [f"size {batch['size']}"] if batch.get("size") else []
    if batch.get("eval_size"):
        parts.append(f"eval {batch['eval_size']}")
    if batch.get("drop_last"):
        parts.append("drop last")
    return ", ".join(parts)


def batch_node(graph, starts, batch):
    made = graph.node("batch", "batch", "batch", batch_text(batch))
    graph.join(starts, made)
    return made


def feed_node(graph, starts, feed, windows=None):
    label = alias_of(feed, "value") or type(feed).__name__
    count = sum(windows.values()) if windows else None
    made = graph.node("feed", "feed", label, params_of(feed), count)
    graph.join(starts, made)
    return made


def tensor_names(data):
    found = {}
    for name in list(getattr(data, "inputs", None) or ()):
        found.setdefault(str(name), "input")
    for name in list(getattr(data, "target_names", None) or ()):
        found.setdefault(str(name), "target")
    return found


def generic(data):
    reporter = getattr(data, "report", None)
    report = reporter() if callable(reporter) else None
    rows = report.get("rows") if isinstance(report, dict) else None
    graph = Flow()
    source = graph.node("source", "source", type(data).__name__, kind_of(data))
    made = set_nodes(graph, source, rows if isinstance(rows, dict) else {})
    tensors = [graph.step(source, f"tensor:{name}", "tensor", name, role) for name, role in tensor_names(data).items()]
    batch_node(graph, (list(made.values()) or [source]) + tensors, getattr(data, "batch", None))
    return graph.result()


def pipeline_of(data):
    method = getattr(data, "pipeline", None)
    found = method() if callable(method) else None
    if not isinstance(found, dict) or not isinstance(found.get("nodes"), list):
        found = generic(data)
    return jsonable({"nodes": found.get("nodes") or [], "edges": found.get("edges") or [],
                     "groups": found.get("groups") or []})
