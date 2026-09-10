import re
import shutil


ANSI = re.compile(r"\x1b\[[0-9;]*m")


class Plain:
    def __getattr__(self, name):
        return lambda text: text


PLAIN = Plain()


def visible(text):
    return ANSI.sub("", str(text))


def wide(text):
    return len(visible(text))


def ljust(text, size):
    return str(text) + " " * max(0, size - wide(text))


ARROW = "─→"
DOT = "·"


def width_of():
    return max(40, shutil.get_terminal_size((96, 24)).columns)


def count(value):
    if value is None:
        return "?"
    return f"{int(value):,}".replace(",", " ")


def short(uri):
    if isinstance(uri, str) and uri.startswith("/"):
        return uri.rsplit("/", 1)[-1]
    return "" if uri is None else str(uri)


def number(value, style=PLAIN):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, dict) and "uri" in value:
        return call_text(value, style=style)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(number(item, style) for item in value) + "]"
    return str(value)


def pad(value, size):
    return ljust(value, size) if wide(value) < size else str(value) + "  "


def params_text(params, skip=(), style=PLAIN):
    parts = []
    for name, value in (params or {}).items():
        if name in skip or value is None:
            continue
        key = style.dim(f"{name}=")
        if isinstance(value, dict) and "uri" in value:
            parts.append(key + call_text(value, style=style))
        elif isinstance(value, dict):
            parts.append(key + "{" + params_text(value, style=style) + "}")
        else:
            parts.append(key + number(value, style))
    return " ".join(parts)


def call_text(call, skip=(), style=PLAIN):
    if call is None:
        return "none"
    if isinstance(call, str):
        return style.cyan(short(call))
    if not isinstance(call, dict):
        return number(call, style)
    if "uri" not in call:
        return params_text(call, skip, style) or "none"
    body = params_text(call.get("params"), skip, style)
    return f"{style.cyan(short(call['uri']))} {body}".rstrip()


def columns_text(names, limit=4):
    names = list(names)
    if not names:
        return "—"
    if len(names) <= limit:
        return ", ".join(names)
    return f"{names[0]} … {names[-1]}  ({len(names)})"


def head(title, width, style):
    return style.bold("── " + title + " " + "─" * max(3, width - len(title) - 4))


def field_line(label, value, style, indent="  ", label_width=12):
    return f"{indent}{style.dim(label.ljust(label_width))}{value}"


def pairs_block(items, style, width, indent="    ", gap=4):
    if not items:
        return []
    name_width = max(wide(name) for name, _ in items)
    value_width = max(wide(value) for _, value in items)
    cell = name_width + 2 + value_width
    columns = max(1, min(3, (width - len(indent) + gap) // (cell + gap)))
    if len(items) <= 4:
        columns = 1
    rows = (len(items) + columns - 1) // columns
    lines = []
    for row in range(rows):
        parts = []
        for column in range(columns):
            position = column * rows + row
            if position >= len(items):
                continue
            name, value = items[position]
            parts.append(ljust(style.dim(ljust(name, name_width)) + "  " + value, cell))
        lines.append(indent + (" " * gap).join(parts).rstrip())
    return lines


def clip(text, limit):
    if wide(text) <= limit:
        return str(text)
    return visible(text)[:max(1, limit - 1)] + "…"


def table(headers, rows, style, indent="  ", width=None):
    if not rows:
        return []
    widths = [max(wide(row[position]) for row in [headers, *rows]) for position in range(len(headers))]
    if width is not None:
        total = sum(widths) + 2 * (len(headers) - 1) + len(indent)
        while total > width:
            longest = max(range(len(widths)), key=lambda position: widths[position])
            if widths[longest] <= 18:
                break
            take = min(total - width, widths[longest] - 18)
            widths[longest] -= take
            total -= take
        rows = [[clip(cell, widths[position]) for position, cell in enumerate(row)] for row in rows]
    def row_text(row):
        cells = [row[position] if wide(row[position]) else "" for position in range(len(headers))]
        return "  ".join(ljust(cell, widths[position]) for position, cell in enumerate(cells)).rstrip()
    span = sum(widths) + 2 * (len(headers) - 1)
    lines = [indent + style.dim(row_text(headers)), indent + style.dim("─" * span)]
    lines.extend(indent + row_text(row) for row in rows)
    return lines
