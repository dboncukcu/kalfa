from . import ALL_SECTIONS, DEFAULT_SECTIONS
from .after import after_section
from .columns import columns_section
from .data import data_section, measure_text
from .model import model_section
from .summary import summary_section
from .text import head, wide, width_of
from .training import training_section
from .wiring import wiring_section


SECTION_TABLE = {
    "summary": (None, summary_section),
    "data": ("DATA", data_section),
    "model": ("MODEL", model_section),
    "training": ("TRAINING", training_section),
    "after": ("AFTER", after_section),
    "columns": ("COLUMNS", columns_section),
    "wiring": ("WIRING", wiring_section),
}


def report(prepared, style, sections=None, probe=None, span=100_000):
    text = render(prepared, style, sections, probe, width=span)
    content = [wide(line) for line in text.splitlines() if wide(line) != span]
    return render(prepared, style, sections, probe, width=max(40, max(content, default=80)))


def render(prepared, style, sections=None, probe=None, width=None):
    width = width or width_of()
    chosen = [name for name in ALL_SECTIONS if name in (sections or DEFAULT_SECTIONS)]
    lines = []
    for name in chosen:
        entry = SECTION_TABLE.get(name)
        if entry is None:
            continue
        title, render_section = entry
        if title is not None:
            lines.append("")
            lines.append(head(title, width, style))
        body = render_section(prepared, style, width, probe)
        lines.extend(body)
    return "\n".join(lines).rstrip("\n") + "\n"


__all__ = ["measure_text", "render", "report"]
