from .document import block_params, group_of
from .text import ARROW, call_text, field_line, pad, short


def after_section(prepared, style, width, probe=None):
    document = prepared.document
    if document is None:
        return [style.dim("  the config could not be shaped, no after analysis")]
    params = block_params(prepared, "after")
    lines = []
    lines.append(field_line("report", f"{pad(params.get('report') or '—', 12)}{style.dim('predict')} "
                                      f"{params.get('predicts') or '—'} {ARROW} predictions.parquet", style))
    plots = group_of(document, "plots")
    if plots:
        named = [name if short((call or {}).get("uri")) == name else f"{name} ({call_text(call, style=style)})"
                 for name, call in plots.items()]
        lines.append(field_line("plots", ", ".join(named) + f"  {ARROW} plots/", style))
    figures = prepared.surface.data.get("figures")
    if figures:
        lines.append(field_line("figures", ", ".join(f"{key} {value}" for key, value in figures.items()), style))
    calibrations = group_of(document, "calibrate")
    if calibrations:
        lines.append(field_line("calibrate", ", ".join(f"{name} ({call_text(call, style=style)})"
                                                       for name, call in calibrations.items())
                                             + f"  {ARROW} fitted/calibrate/", style))
    generate = params.get("generate")
    if generate:
        lines.append(field_line("generate", f"{call_text(generate, style=style)}  {ARROW} samples/", style))
    record = prepared.surface.data.get("record")
    if record is not None:
        lines.append(field_line("record", str(record), style))
    return lines
