import base64
import html
import io
import math

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from kalfa import summary as summarizing
from kalfa.std.common import numeric
from kalfa.std.figure import PALETTE


DPI = 100
CSS = """
:root { color-scheme: light; --ink: #16181d; --muted: #6b7079; --line: #dfe2e7; --soft: #f4f6f9; --accent: #2a78d6; }
* { box-sizing: border-box; }
body { margin: 0; background: #fff; color: var(--ink);
       font: 14px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 28px 16px 64px; }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 19px; margin: 36px 0 10px; padding-top: 10px; border-top: 1px solid var(--line); }
h3 { font-size: 16px; margin: 30px 0 4px; }
h4 { font-size: 14px; margin: 16px 0 6px; }
h5 { font-size: 13px; margin: 14px 0 6px; }
p { margin: 6px 0; overflow-wrap: anywhere; }
nav a { color: var(--accent); text-decoration: none; margin-right: 14px; }
.muted { color: var(--muted); }
.best { font-weight: 600; }
.note { background: #fff8e6; border: 1px solid #f1d8a0; border-radius: 6px; padding: 6px 10px; }
.scroll { overflow-x: auto; margin: 6px 0 14px; }
table { border-collapse: collapse; font-size: 12.5px; }
th, td { padding: 3px 9px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }
th { background: var(--soft); font-weight: 600; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.pair { display: flex; flex-wrap: wrap; gap: 0 28px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 420px), 1fr)); gap: 0 18px; }
figure { margin: 10px 0 18px; }
figure img { display: block; max-width: 100%; height: auto; border: 1px solid var(--line); border-radius: 6px; }
figcaption { color: var(--muted); font-size: 12.5px; margin-top: 4px; }
section.record { margin-bottom: 18px; }
@media print { figure, table { break-inside: avoid; } nav { display: none; } }
"""


def escape(value):
    return html.escape(str(value), quote=True)


def finite(value):
    return numeric(value) and math.isfinite(float(value))


def shown(value, column=None):
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(shown(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key}: {shown(item)}" for key, item in value.items()) + "}"
    return summarizing.cell(value, column)


def plural(count, word):
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def paragraph(text, kind=None):
    attribute = f' class="{kind}"' if kind else ""
    return f"<p{attribute}>{escape(text)}</p>"


def table(headers, rows):
    head = "".join(f"<th>{escape(header)}</th>" for header in headers)
    lines = []
    for row in rows:
        cells = []
        for value, header in zip(row, headers):
            kind = ' class="num"' if numeric(value) else ""
            cells.append(f"<td{kind}>{escape(shown(value, header))}</td>")
        lines.append("<tr>" + "".join(cells) + "</tr>")
    body = "".join(lines)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def new_figure(width=6.0, height=3.6, columns=1):
    figure = Figure(figsize=(width, height), layout="constrained")
    FigureCanvasAgg(figure)
    return figure, figure.subplots(1, columns)


def embedded(figure, caption):
    with io.BytesIO() as buffer:
        figure.savefig(buffer, format="png", dpi=DPI, metadata={"Software": None})
        data = buffer.getvalue()
    figure.clear()
    encoded = base64.b64encode(data).decode("ascii")
    return (f'<figure><img alt="{escape(caption)}" src="data:image/png;base64,{encoded}">'
            f"<figcaption>{escape(caption)}</figcaption></figure>")


def grid(figures):
    return '<div class="grid">' + "".join(figures) + "</div>"


def color_of(position):
    return PALETTE[position % len(PALETTE)]


def document(title, body):
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f"<title>{escape(title)}</title>\n<style>{CSS}</style>\n</head>\n<body>\n<main>\n{body}\n</main>\n"
            "</body>\n</html>\n")
