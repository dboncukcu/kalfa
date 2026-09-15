from ..std.pre.base import matches_any
from .document import batch_size, data_params, field_plan, owners_of, spectators_of
from .text import (ARROW, DOT, PLAIN, call_text, columns_text, count, field_line, number, pad, params_text, short,
                   table, width_of)


def split_text(split, style=PLAIN):
    if not isinstance(split, dict):
        return number(split, style)
    params = split.get("params") if "uri" in split else split
    params = params or {}
    name = style.cyan(short(split.get("uri")) or "random")
    ratios = params.get("ratios")
    body = params_text(params, skip=("ratios",), style=style)
    text = name
    if isinstance(ratios, list):
        text += "  " + " / ".join(f"{float(ratio):g}" for ratio in ratios)
    return f"{text}  {body}".rstrip()


def batch_text(size, style):
    return number(size, style) if size is not None else style.dim("the whole set")


def transform_texts(params, style):
    before = [call_text(item, style=style) for item in params.get("transform_pre") or []]
    after = []
    for name, entry in (params.get("set_transforms") or {}).items():
        for item in entry.get("transforms") or []:
            after.append(f"{call_text(item, style=style)} ({name})")
    return before, after


def sizes_line(sizes, style, sets):
    if sizes is None:
        return style.dim("sizes unknown")
    return f"  {DOT}  ".join(f"{name} {count(sizes.get(name))}" for name in sets)


def data_section(prepared, style, width, probe=None):
    params = data_params(prepared)
    if params is None:
        return [style.dim("  the config could not be shaped, no data analysis")]
    header = prepared.header
    lines = []
    source = params.get("source") or {}
    shape = ""
    if header is not None:
        shape = f"{count(header['rows'])} rows {DOT} {len(header['columns'])} columns"
    path = ((source.get("params") or {}).get("path"))
    text = f"{style.cyan(short(source.get('uri')))}  {path}" if path else call_text(source, style=style)
    lines.append(field_line("source", f"{pad(text, 44)}{shape}", style))
    sizes = (probe.sizes if probe is not None and probe.sizes else None) or prepared.sizes
    lines.append(field_line("split", f"{pad(split_text(params.get('split'), style), 44)}"
                                     f"{sizes_line(sizes, style, prepared.sets)}", style))
    size = batch_size(params)
    lines.append(field_line("batch", f"{pad(batch_text(size, style), 44)}{style.dim('feed')}  "
                                    f"{call_text(params.get('feed'), style=style)}", style))
    before, after = transform_texts(params, style)
    if before or after:
        text = ", ".join(before) or style.dim("none before the split")
        if after:
            text += f"  {DOT}  after the split: " + ", ".join(after)
        if probe is None:
            text += "  " + style.dim("the sizes above are from the header, before these; --measure counts them")
        lines.append(field_line("transforms", text, style))
    frames = ((params.get("frames") or {}).get("params") or {}).get("frames") or []
    if frames:
        lines.append(field_line("frames", ", ".join(call_text(item, style=style) for item in frames)
                                          + f"  {style.dim('fitted on train')}", style))
    _, drop, _ = field_plan(prepared)
    if drop:
        lines.append(field_line("drop", ", ".join(str(name) for name in drop), style))
    if params.get("mask") is not None:
        lines.append(field_line("mask", f"{params['mask']}  {style.dim('kept in the frame, not scored')}", style))
    lines.append("")
    lines.extend(fields_table(prepared, style, width))
    lines.append("")
    lines.extend(data_tree(prepared, params, sizes, style, probe, prepared.sets))
    return lines


def fields_table(prepared, style, width=None):
    params = data_params(prepared) or {}
    fields, _, _ = field_plan(prepared)
    if not fields:
        return []
    owners, _ = owners_of(prepared)
    keys = params.get("preprocessors_keys") or {}
    rows = []
    for pattern, spec in fields.items():
        spec = spec or {}
        matched = [column for column, owner in owners.items() if owner == pattern]
        chain = []
        for name in spec.get("preprocessors") or []:
            sets = ((keys.get(name) or {}).get("sets"))
            chain.append(f"{name} ({', '.join(sets)} only)" if sets else name)
        rows.append([pattern, columns_text(matched), f" {ARROW} ".join(chain) or "—",
                     "target" if spec.get("target") else "feature"])
    _, columns = owners_of(prepared)
    for pattern in spectators_of(prepared):
        matched = [column for column in columns if column not in owners and matches_any(column, [pattern])]
        rows.append([pattern, columns_text(matched), "—", "spectator"])
    return table(["field", "columns", "preprocessors", "role"], rows, style, width=width)


def data_tree(prepared, params, sizes, style, probe, sets):
    source = params.get("source") or {}
    path = ((source.get("params") or {}).get("path")) or call_text(source, style=style)
    feed = short((params.get("feed") or {}).get("uri")) or "feed"
    _, _, names = field_plan(prepared)
    fitted = ", ".join(names)
    lines = [f"  {style.bold(str(path))} {ARROW} split "
             f"{style.cyan(short((params.get('split') or {}).get('uri')) or 'random')}"]
    shape = ""
    if probe is not None and probe.features is not None:
        size = batch_size(params)
        shape = f" {ARROW} x [{batch_text(size, style)}, {probe.features}]"
    steps = {}
    for name in sets:
        steps[name] = "no preprocessors" if not names else (f"fit {fitted}" if name == "train" else "apply")
    span = max(len(step) for step in steps.values())
    for position, name in enumerate(sets):
        corner = "└─" if position == len(sets) - 1 else "├─"
        lines.append(f"    {corner} {name.ljust(5)} {count((sizes or {}).get(name)):>8}  {ARROW} "
                     f"{steps[name].ljust(span)} {ARROW} {feed}{shape}")
    return lines


def measure_text(prepared, style, sizes=None):
    params = data_params(prepared) or {}
    sizes = sizes if sizes is not None else prepared.measured
    source = (((params.get("source") or {}).get("params") or {}).get("path")
              or call_text(params.get("source"), style=style))
    rows = f" {count(prepared.header['rows'])} rows" if prepared.header is not None else ""
    names = ", ".join(field_plan(prepared)[2])
    before, after = transform_texts(params, style)
    transforms = len(before) + len(after)
    size = batch_size(params)
    steps = [f"{source}{rows}"]
    if transforms:
        steps.append(f"{transforms} transforms")
    steps.append(f"split {split_text(params.get('split'), style)}")
    steps.append(f"fitted {names} on train" if names else "no preprocessors")
    steps.append(f"{short((params.get('feed') or {}).get('uri')) or 'feed'} feed")
    steps.append(f"3 loaders, batch {number(size, style)}")
    lines = ["measured the data block:"]
    limit = width_of()
    for step in steps:
        piece = f"{ARROW} {step}" if lines[-1] != "measured the data block:" else step
        if len(lines[-1]) + len(piece) + 3 > limit:
            lines.append(f"  {ARROW} {step}")
        else:
            lines[-1] += f"  {piece}" if lines[-1] == "measured the data block:" else f" {piece}"
    lines.append(f"sets after filters: {sizes_line(sizes, style, prepared.sets)}")
    return "\n".join(lines)
