import io
import json

from ruamel.yaml import YAML


WIDTH = 88
SHORT = 150
FLOW = set(",[]{}")
GLOB = set("*?")
DUMPER = YAML(typ="safe")
DUMPER.width = 4096


def flat(text):
    return " ".join(str(text or "").split())


def short(text):
    text = flat(text)
    return text.split("; ")[0] if len(text) > SHORT and "; " in text else text


def lower_first(text):
    return text[0].lower() + text[1:] if len(text) > 1 and text[0].isupper() and text[1].islower() else text


def and_list(items):
    items = [str(item) for item in items]
    if len(items) < 2:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def unique(items):
    return list(dict.fromkeys(items))


def plain(text):
    stream = io.StringIO()
    DUMPER.dump(text, stream)
    out = stream.getvalue()
    for tail in ("\n...\n", "\n"):
        if out.endswith(tail):
            return out[:-len(tail)] == text
    return False


def scalar_text(value, inner=False):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value != value:
        return ".nan"
    if isinstance(value, float) and value in (float("inf"), float("-inf")):
        return ".inf" if value > 0 else "-.inf"
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value)
    if text and plain(text) and not GLOB & set(text) and not (inner and (FLOW & set(text) or ": " in text)):
        return text
    return json.dumps(text, ensure_ascii=False)


def flow(value):
    if isinstance(value, dict):
        return "{" + ", ".join(f"{scalar_text(key, True)}: {flow(item)}" for key, item in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(flow(item) for item in value) + "]"
    return scalar_text(value, True)


def value_text(value):
    return value if isinstance(value, str) else flow(value)


def yaml_lines(key, value, indent=""):
    head = f"{indent}{scalar_text(key)}:"
    one = f"{head} {flow(value)}"
    if not isinstance(value, (dict, list)) or not value or len(one) <= WIDTH:
        return [one]
    if isinstance(value, dict):
        return [head] + [line for inner, item in value.items() for line in yaml_lines(inner, item, indent + "  ")]
    return [head] + [line for item in value for line in item_lines(item, indent + "  ")]


def item_lines(item, indent):
    one = f"{indent}- {flow(item)}"
    if len(one) <= WIDTH or not isinstance(item, dict) or not item:
        return [one]
    lines = [line for key, value in item.items() for line in yaml_lines(key, value, indent + "  ")]
    return [f"{indent}- {lines[0][len(indent) + 2:]}"] + lines[1:]


def yaml_text(key, value):
    return "\n".join(yaml_lines(key, value))
