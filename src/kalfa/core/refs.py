import re

from .errors import RunError


TABLES = ("models", "losses", "metrics", "optimizers")
SINGULAR = {"models": "model", "losses": "loss", "metrics": "metric", "optimizers": "optimizer"}
SEGMENT = re.compile(r"(?P<name>[^.\[\]\s]+)(?:\[\d+\])*")
INDEX = re.compile(r"\[(\d+)\]")


def tokenize(path):
    if not isinstance(path, str) or not path.strip():
        raise RunError(f"a path is dotted like optimizers.main.lr, got {path!r}")
    tokens = []
    for piece in path.strip().split("."):
        match = SEGMENT.fullmatch(piece)
        if match is None:
            raise RunError(f"malformed path {path!r}: {piece!r} is not a name with an optional [index]")
        tokens.append(match.group("name"))
        tokens.extend(int(index) for index in INDEX.findall(piece))
    return tokens


def join_tokens(tokens):
    text = ""
    for token in tokens:
        if isinstance(token, int):
            text += f"[{token}]"
        else:
            text += f".{token}" if text else token
    return text


def ref_parts(asked):
    if isinstance(asked, str) and asked.startswith("@"):
        return asked[1:]
    if isinstance(asked, dict) and "ref" in asked:
        attr = asked.get("attr")
        return f"{asked['ref']}.{attr}" if attr else str(asked["ref"])
    return None


def ref_json(asked):
    tokens = tokenize(ref_parts(asked))
    split = 2 if tokens[0] in TABLES else 1
    out = {"ref": join_tokens(tokens[:split])}
    if tokens[split:]:
        out["attr"] = join_tokens(tokens[split:])
    return out


def ref_json_of(ctx, value):
    for section in TABLES:
        for name, entry in getattr(ctx, section).items():
            if entry is value or getattr(entry, "module", None) is value:
                return {"ref": f"{section}.{name}"}
            if section == "models" and getattr(entry, "ema", None) is value:
                return {"ref": f"{section}.{name}", "attr": "ema"}
    for section in ("data", "generate"):
        if getattr(ctx, section, None) is value:
            return {"ref": section}
    for node in ctx.tree.nodes.values():
        if node.obj is value:
            return {"ref": node.path}
    return None


def jsonable(value, ctx=None):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    from .build import Node, Ref

    if isinstance(value, Ref):
        return {"ref": value.path} if value.attr is None else {"ref": value.path, "attr": value.attr}
    if isinstance(value, Node):
        return {"node": value.path}
    if isinstance(value, dict):
        return {str(key): jsonable(item, ctx) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item, ctx) for item in value]
    if ctx is not None:
        found = ref_json_of(ctx, value)
        if found is not None:
            return found
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)
