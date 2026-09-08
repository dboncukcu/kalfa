import re
import shutil
from pathlib import Path

from .kinds import SETS
from .std.pre import assign_fields

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

DEFAULT_SECTIONS = ("summary", "data", "model", "training", "after", "columns")
ALL_SECTIONS = (*DEFAULT_SECTIONS, "wiring")
ARROW = "─→"
DOT = "·"


def width_of():
    """The width the terminal reports: COLUMNS overrides it, a pipe has none and falls back to 96 columns."""
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


def data_params(prepared):
    document = prepared.document
    if document is None:
        return None
    return ((document.get("flow") or {}).get("data") or {}).get("params") or {}


def block_params(prepared, name):
    document = prepared.document
    if document is None:
        return {}
    return ((document.get("flow") or {}).get(name) or {}).get("params") or {}


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


def sizes_line(sizes, style):
    if sizes is None:
        return style.dim("sizes unknown")
    return f"  {DOT}  ".join(f"{name} {count(sizes.get(name))}" for name in SETS)


def owners_of(prepared):
    params = data_params(prepared) or {}
    header = prepared.header
    if header is None:
        return {}, []
    drop = params.get("drop") or []
    columns = [name for name in header["columns"] if name not in drop]
    fields = params.get("fields") or {}
    owners, _ = assign_fields(columns, list(fields))
    return owners, columns


def target_fields(prepared, probe=None):
    """The target fields in the order the plan builds them: from the fitted plan when the data was loaded, else
    from the file header and the field patterns, pattern by pattern, column by column."""
    if probe is not None and probe.prep is not None:
        return [item.name for item in probe.prep.fields if item.target]
    params = data_params(prepared) or {}
    fields = params.get("fields") or {}
    owners, columns = owners_of(prepared)
    found = []
    for pattern, spec in fields.items():
        if (spec or {}).get("target"):
            found.extend([column for column in columns if owners.get(column) == pattern])
    return found


def target_slots(prepared, probe=None):
    """The place every target field takes in the output wire that predicts it, from training.targets."""
    from .std.runtime import expand_targets

    mapping = block_params(prepared, "after").get("targets") or {}
    fields = target_fields(prepared, probe)
    slots = {}
    for wire, selector in mapping.items():
        for position, name in enumerate(expand_targets(selector, fields)):
            slots[name] = (wire, position)
    return slots


def summary_section(prepared, style, width, probe=None):
    surface = prepared.surface
    lines = []
    from cirak.registry import registry

    from . import __version__

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
    lines.append(field_line("split", f"{pad(split_text(params.get('split'), style), 44)}{sizes_line(sizes, style)}",
                            style))
    batch = params.get("batch") or {}
    size = batch.get("size") if isinstance(batch, dict) else batch
    lines.append(field_line("batch", f"{pad(number(size, style), 44)}{style.dim('feed')}  "
                                    f"{call_text(params.get('feed'), style=style)}", style))
    filters = list(params.get("filter_pre") or []) + list(params.get("filter_set") or [])
    if filters:
        lines.append(field_line("filters", ", ".join(call_text(item, style=style) for item in filters), style))
    drop = params.get("drop") or []
    if drop:
        lines.append(field_line("drop", ", ".join(str(name) for name in drop), style))
    lines.append("")
    lines.extend(fields_table(prepared, style, width))
    lines.append("")
    lines.extend(data_tree(prepared, params, sizes, style, probe))
    return lines


def fields_table(prepared, style, width=None):
    params = data_params(prepared) or {}
    fields = params.get("fields") or {}
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
    return table(["field", "columns", "preprocessors", "role"], rows, style, width=width)


def data_tree(prepared, params, sizes, style, probe=None):
    source = params.get("source") or {}
    path = ((source.get("params") or {}).get("path")) or call_text(source, style=style)
    feed = short((params.get("feed") or {}).get("uri")) or "feed"
    names = [name for name in (params.get("preprocessors") or {})]
    fitted = ", ".join(names)
    lines = [f"  {style.bold(str(path))} {ARROW} split "
             f"{style.cyan(short((params.get('split') or {}).get('uri')) or 'random')}"]
    shape = ""
    if probe is not None and probe.features is not None:
        batch = params.get("batch") or {}
        size = batch.get("size") if isinstance(batch, dict) else batch
        shape = f" {ARROW} x [{number(size, style)}, {probe.features}]"
    steps = {}
    for name in SETS:
        steps[name] = "no preprocessors" if not names else (f"fit {fitted}" if name == "train" else "apply")
    span = max(len(step) for step in steps.values())
    for position, name in enumerate(SETS):
        corner = "└─" if position == len(SETS) - 1 else "├─"
        lines.append(f"    {corner} {name.ljust(5)} {count((sizes or {}).get(name)):>8}  {ARROW} "
                     f"{steps[name].ljust(span)} {ARROW} {feed}{shape}")
    return lines


def load_text(prepared, style, sizes=None):
    params = data_params(prepared) or {}
    sizes = sizes if sizes is not None else prepared.loaded
    source = (((params.get("source") or {}).get("params") or {}).get("path")
              or call_text(params.get("source"), style=style))
    rows = f" {count(prepared.header['rows'])} rows" if prepared.header is not None else ""
    names = ", ".join(params.get("preprocessors") or {})
    filters = len(list(params.get("filter_pre") or []) + list(params.get("filter_set") or []))
    batch = params.get("batch") or {}
    size = batch.get("size") if isinstance(batch, dict) else batch
    steps = [f"{source}{rows}"]
    if filters:
        steps.append(f"{filters} filters")
    steps.append(f"split {split_text(params.get('split'), style)}")
    steps.append(f"fitted {names} on train" if names else "no preprocessors")
    steps.append(f"{short((params.get('feed') or {}).get('uri')) or 'feed'} feed")
    steps.append(f"3 loaders, batch {number(size, style)}")
    lines = ["loaded the data block:"]
    limit = width_of()
    for step in steps:
        piece = f"{ARROW} {step}" if lines[-1] != "loaded the data block:" else step
        if len(lines[-1]) + len(piece) + 3 > limit:
            lines.append(f"  {ARROW} {step}")
        else:
            lines[-1] += f"  {piece}" if lines[-1] == "loaded the data block:" else f" {piece}"
    lines.append(f"sets after filters: {sizes_line(sizes, style)}")
    return "\n".join(lines)


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
        return f"{name} {number(params['in_features'], style)}→{number(params['out_features'], style)} {rest}".rstrip()
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


TRIGGER_TEXT = {
    "/trigger/kalfa/after_turn": lambda p: f"turn ≥ {number(p.get('at'))}",
    "/trigger/kalfa/metric_below": lambda p: f"{p.get('monitor')} < {number(p.get('value'))}",
    "/trigger/kalfa/metric_above": lambda p: f"{p.get('monitor')} > {number(p.get('value'))}",
    "/trigger/kalfa/plateau": lambda p: f"{p.get('monitor')} plateau {number(p.get('patience'))}",
    "/trigger/kalfa/time_budget": lambda p: f"after {number(p.get('minutes'))} minutes",
}


def trigger_text(call, style=PLAIN):
    if not isinstance(call, dict):
        return str(call)
    render = TRIGGER_TEXT.get(call.get("uri"))
    params = call.get("params") or {}
    if render is None:
        return call_text(call, style=style)
    return render(params)


def unwrap(call):
    if isinstance(call, dict) and short(call.get("uri")) in ("criterion", "metric"):
        inner = (call.get("params") or {}).get("criterion") or (call.get("params") or {}).get("metric")
        if isinstance(inner, dict):
            return inner
    return call


def group_of(document, name):
    return (document or {}).get(name) or {}


def reference(text):
    return text.split(".", 1)[-1] if isinstance(text, str) and text.startswith("@") else text


def sets_of(keys, style=PLAIN):
    keys = keys or {}
    sets = keys.get("sets") or list(SETS)
    text = ", ".join(sets)
    every = keys.get("every")
    if every:
        text += style.dim(f"  every {number(every, style)}")
    return text


def compares_of(keys, style=PLAIN):
    """The wire and the target fields a definition compares, when it names either."""
    keys = keys or {}
    output, target = keys.get("output"), keys.get("target")
    if output is None and target is None:
        return ""
    selector = target if isinstance(target, str) else (", ".join(target) if target else "")
    return f"{output or '—'} {ARROW} {selector or '—'}"


def definition_table(label, definitions, keys_table, style, width, notes):
    if not definitions:
        return []
    keys_table = keys_table or {}
    rows = []
    for name, call in definitions.items():
        keys = keys_table.get(name) or {}
        rows.append([name, call_text(unwrap(call), style=style), compares_of(keys, style), sets_of(keys, style),
                     style.dim(notes.get(name, ""))])
    if any(row[2] for row in rows):
        headers = [label, "lego", "compares", "reported on", ""]
    else:
        rows = [[row[0], row[1], row[3], row[4]] for row in rows]
        headers = [label, "lego", "reported on", ""]
    return ["", *table(headers, rows, style, width=width)]


def training_section(prepared, style, width, probe=None):
    document = prepared.document
    if document is None:
        return [style.dim("  the config could not be shaped, no training analysis")]
    params = block_params(prepared, "training")
    optimizers = block_params(prepared, "optimizers").get("optimizer_items") or []
    lines = []
    horizon = f"{number(params.get('epochs'), style)} epochs" if params.get("epochs") is not None else \
        f"{number((params.get('steps') or {}).get('total'), style)} steps of " \
        f"{number((params.get('steps') or {}).get('turn'), style)}"
    turn = params.get("turn") or {}
    extra = " ".join(part for part in (params_text(turn.get("params"), style=style),
                                       params_text(params.get("turn_params"), style=style)) if part)
    lines.append(field_line("turn", f"{pad(style.cyan(short(turn.get('uri'))), 20)}{pad(horizon, 16)}"
                                    f"{style.dim('predicts')} {params.get('predicts') or '—'}", style))
    if extra:
        lines.append(field_line("", extra, style))
    from .std.runtime import expand_targets

    mapping = block_params(prepared, "after").get("targets") or {}
    if mapping:
        fields = target_fields(prepared, probe)
        rows = [[wire, ARROW, ", ".join(expand_targets(selector, fields)) or str(selector)]
                for wire, selector in mapping.items()]
        lines.append("")
        lines.extend(table(["output wire", "", "predicts the target fields"], rows, style, width=width))
    rows = []
    for item in optimizers:
        rows.append([item["name"], call_text({"uri": item["uri"], "params": item.get("params")}, style=style),
                     ARROW + " " + ", ".join((item.get("models") or {}).values()), item.get("loss") or "—",
                     call_text(item.get("schedule"), style=style) if item.get("schedule")
                     else style.dim("no schedule")])
    if rows:
        lines.append("")
        lines.extend(table(["optimizer", "lego", "trains", "loss", "schedule"], rows, style, width=width))
    active = {item.get("loss") for item in optimizers}
    losses = group_of(document, "losses")
    notes = {name: "active at turn 1" if name in active else "held for the rules" for name in losses}
    lines.extend(definition_table("loss", losses, params.get("losses_keys"), style, width, notes))
    lines.extend(definition_table("metric", group_of(document, "metrics"), params.get("metrics_keys"), style,
                                  width, {}))
    lines.append("")
    checkpoint = params.get("checkpoint")
    lines.append(field_line("checkpoint", f"{pad(call_text(checkpoint, style=style) if checkpoint else 'none', 44)}"
                                          f"{style.dim('report')} "
                                          f"{(block_params(prepared, 'after') or {}).get('report')}", style))
    triggers = group_of(document, "triggers")
    stop = params.get("stop") or []
    if stop:
        lines.append(field_line("stop", ", ".join(trigger_text(triggers.get(reference(item)), style)
                                                  for item in stop), style))
    rules = params.get("rules") or []
    if rules:
        lines.append("")
        lines.append("  " + style.dim("rules (evaluated at the end of a turn, effective in the next)"))
        rows = []
        for rule in rules:
            when = trigger_text(triggers.get(reference(rule.get("when"))), style)
            sets = ", ".join(f"{key} := {value}" for key, value in (rule.get("set") or {}).items())
            after = style.dim(f"after {rule['after']}") if rule.get("after") else ""
            rows.append([when, ARROW, rule.get("name"), sets, after])
        lines.extend(table(["", "", "", "", ""], rows, style, indent="    ", width=width)[2:])
    return lines


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
    generate = params.get("generate")
    if generate:
        lines.append(field_line("generate", f"{call_text(generate, style=style)}  {ARROW} samples/", style))
    record = prepared.surface.data.get("record")
    if record is not None:
        lines.append(field_line("record", str(record), style))
    return lines


def column_refs(prepared):
    from cirak.registry import registry

    from .check import Checker

    found = {}
    try:
        for column, path in Checker(prepared.surface, registry).column_refs():
            found[column] = ".".join(str(part) for part in path[1:-2])
    except Exception:
        return {}
    return found


def columns_section(prepared, style, width, probe=None):
    header = prepared.header
    if header is None:
        if probe is not None and probe.prep is not None:
            return plan_columns(prepared, style, width, probe)
        return [style.dim("  the data header could not be read (a source lego of your own reads it), "
                          "no column table; --load builds it from the fitted plan")]
    params = data_params(prepared) or {}
    fields = params.get("fields") or {}
    drop = list(params.get("drop") or [])
    owners, _ = owners_of(prepared)
    refs = column_refs(prepared)
    wires = target_slots(prepared, probe)
    produced = {}
    slots = {}
    if probe is not None and probe.prep is not None:
        features = list(probe.prep.features)
        for item in probe.prep.fields:
            produced[item.name] = list(item.columns)
            positions = [features.index(name) for name in item.columns if name in features]
            if positions:
                slots[item.name] = (min(positions), max(positions))
    rows = []
    for column in header["columns"]:
        dtype = str(header["dtypes"].get(column, "?"))
        pattern = owners.get(column)
        spec = (fields.get(pattern) or {}) if pattern is not None else {}
        chain = f" {ARROW} ".join(spec.get("preprocessors") or []) or "—"
        width_note = produced.get(column)
        if width_note and len(width_note) > 1:
            chain += f"  ({len(width_note)} columns)"
        if column in drop:
            rows.append([column, dtype, "—", "—", "dropped", "—"])
        elif pattern is None:
            role = f"read by {refs[column]}" if column in refs else "not a field"
            rows.append([column, dtype, "—", "—", role, "—"])
        elif spec.get("target"):
            wire, position = wires.get(column, (None, None))
            rows.append([column, dtype, pattern, chain, "target",
                         column if wire is None else f"{wire}[{position}]"])
        else:
            span = slots.get(column)
            tensor = "x"
            if span is not None:
                tensor = f"x[{span[0]}]" if span[0] == span[1] else f"x[{span[0]} … {span[1]}]"
            rows.append([column, dtype, pattern, chain, "feature", tensor])
    lines = table(["column", "dtype", "field", "preprocessors", "role", "tensor"], rows, style, width=width)
    if probe is None:
        lines.append("  " + style.dim("the produced widths and the tensor slots need --load"))
    return lines


def plan_columns(prepared, style, width, probe):
    """The column table of a run whose header could not be read: every field of the fitted plan, in plan order."""
    prep = probe.prep
    features = list(prep.features)
    wires = target_slots(prepared, probe)
    rows = []
    for item in prep.fields:
        chain = f" {ARROW} ".join(item.chain) or "—"
        if len(item.columns) > 1:
            chain += f"  ({len(item.columns)} columns)"
        dtype = str(prep.dtypes.get(item.columns[0] if item.columns else item.name, "?"))
        if item.target:
            wire, position = wires.get(item.name, (None, None))
            rows.append([item.name, dtype, chain, "target", item.name if wire is None else f"{wire}[{position}]"])
            continue
        positions = [features.index(name) for name in item.columns if name in features]
        span = f"x[{min(positions)} … {max(positions)}]" if len(positions) > 1 else (
            f"x[{positions[0]}]" if positions else "—")
        rows.append([item.name, dtype, chain, "feature", span])
    return table(["column", "dtype", "preprocessors", "role", "tensor"], rows, style, width=width)


def wiring_section(prepared, style, width, probe=None):
    if not prepared.implicit:
        return [style.dim("  no implicit bindings")]
    return [f"  {path}: {param} ← {key}" for path, param, key in prepared.implicit]


SECTION_TABLE = {
    "summary": (None, summary_section),
    "data": ("DATA", data_section),
    "model": ("MODEL", model_section),
    "training": ("TRAINING", training_section),
    "after": ("AFTER", after_section),
    "columns": ("COLUMNS", columns_section),
    "wiring": ("WIRING", wiring_section),
}


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
