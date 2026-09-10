from .document import block_params
from .text import ARROW, DOT, PLAIN, count, number, params_text, short, table


def spec_text(entry, style=PLAIN):
    if not isinstance(entry, dict):
        return number(entry, style)
    if "block" in entry:
        body = params_text(entry.get("params"), style=style)
        return f"{style.cyan(entry['block'])}({body})" if body else style.cyan(str(entry["block"]))
    if "model" in entry:
        return style.cyan(str(entry["model"]))
    params = entry.get("params") or {}
    name = style.cyan(short(entry.get("uri")))
    if "in_features" in params and "out_features" in params:
        rest = params_text(params, skip=("in_features", "out_features"), style=style)
        width = f"{number(params['in_features'], style)}→{number(params['out_features'], style)}"
        return f"{name} {width} {rest}".rstrip()
    if "out_features" in params:
        rest = params_text(params, skip=("out_features",), style=style)
        return f"{name} {number(params['out_features'], style)} {rest}".rstrip()
    body = params_text(params, style=style)
    return f"{name} {body}".rstrip()


def chain_lines(parts, indent, width):
    lines = []
    current = indent + parts[0]
    for part in parts[1:]:
        piece = f" {ARROW} {part}"
        if width is not None and len(current) + len(piece) > width:
            lines.append(current)
            current = f"{indent}  {ARROW} {part}"
        else:
            current += piece
    lines.append(current)
    return lines


def block_lines(blocks, name, style, indent="    ", width=None):
    block = (blocks or {}).get(name)
    if not isinstance(block, dict):
        return [f"{indent}{style.dim('no block')}"]
    inputs = ", ".join(block.get("inputs") or [])
    outputs = ", ".join(block.get("outputs") or [])
    if "spec" in block:
        parts = [inputs or "x", *[spec_text(entry, style) for entry in block["spec"]], outputs or "y"]
        return chain_lines(parts, indent, width)
    graph = block.get("graph") or {}
    rows = []
    for node, entry in graph.items():
        entry = entry or {}
        wires = ", ".join(entry.get("inputs") or [])
        produced = ", ".join(entry.get("outputs") or [node])
        call = spec_text(entry, style)
        repeat = entry.get("repeat")
        if repeat is not None:
            call += f"  ×{number(repeat, style)}"
        rows.append([wires or "—", ARROW, call, ARROW, produced])
    lines = [f"{indent}{style.dim('inputs ' + (inputs or '?') + '   outputs ' + (outputs or '?'))}"]
    lines.extend(table(["", "", "", "", ""], rows, style, indent=indent, width=width)[2:])
    return lines


def model_section(prepared, style, width, probe=None):
    document = prepared.document
    if document is None:
        return [style.dim("  the config could not be shaped, no model analysis")]
    params = block_params(prepared, "models")
    blocks = document.get("blocks") or {}
    optimizers = {}
    for item in block_params(prepared, "optimizers").get("optimizer_items") or []:
        for model in (item.get("models") or {}).values():
            optimizers.setdefault(model, item["name"])
    emas = {item["name"]: item.get("decay") for item in params.get("ema_items") or []}
    lines = []
    for item in params.get("trained_items") or []:
        name = item["name"]
        facts = []
        facts.append("trained" if item.get("trainable", True) else "frozen (eval mode)")
        facts.append(f"optimizer {optimizers[name]}" if name in optimizers else "no optimizer")
        if item.get("init"):
            facts.append("init " + params_text(item["init"], style=style))
        if item.get("weights"):
            facts.append("weights " + params_text(item["weights"], style=style))
        if name in emas:
            facts.append(f"ema decay {number(emas[name], style)}")
        if probe is not None and name in probe.parameters:
            total, trainable = probe.parameters[name]
            facts.append(f"{count(total)} parameters"
                         + ("" if total == trainable else f" ({count(trainable)} trainable)"))
        elif probe is not None:
            facts.append("parameters after the first batch")
        lines.append(f"  {style.bold(name)}   {style.dim(f'  {DOT}  '.join(facts))}")
        lines.extend(block_lines(blocks, name, style, width=width))
        lines.append("")
    for item in params.get("composite_items") or []:
        name = item["name"]
        lines.append(f"  {style.bold(name)}   {style.dim('composite')}")
        lines.extend(block_lines(blocks, name, style, width=width))
        lines.append("")
    return lines[:-1] if lines and lines[-1] == "" else lines
