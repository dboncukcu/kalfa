from pathlib import Path

from cirak.registry import registry

from .. import __version__
from .text import call_text, field_line, number, pairs_block


def summary_section(prepared, style, width, probe=None):
    surface = prepared.surface
    lines = []

    fragments = {str(path): uri for uri, path in registry.fragments().items()}
    files = [Path(path).name for path in surface.paths]
    included = [fragments.get(loaded.file) or Path(loaded.file).name for loaded in surface.layer.walk()
                if loaded.file != "--set" and Path(loaded.file).name not in files]
    title = ", ".join(files)
    lines.append(style.bold(title) + style.dim(f"{'kalfa ' + __version__:>{max(1, width - len(title))}}"))
    layers = ", ".join(files + included)
    lines.append(field_line("layers", layers, style))
    params = surface.raw.get("params")
    if isinstance(params, dict) and params:
        lines.append("  " + style.dim("params"))
        lines.extend(pairs_block([(name, number(value, style)) for name, value in params.items()], style, width))
    config = surface.data
    seed = config.get("seed")
    device = config.get("device")
    lines.append(field_line("seed", "unseeded" if seed is None else number(seed, style), style))
    lines.append(field_line("device", call_text(device, style=style) if device is not None
                                      else style.dim("cpu (no device key)"), style))
    record = config.get("record")
    if record is not None:
        lines.append(field_line("record", str(record), style))
    return lines
