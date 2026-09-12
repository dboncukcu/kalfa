from cirak.registry import registry

from ..std.common.effects import relative_effect
from ..std.common.runtime import expand_targets
from .document import block_params, group_of, reference, target_fields, unwrap
from .text import ARROW, PLAIN, call_text, field_line, number, pad, params_text, short, table


class Shown(dict):
    def __missing__(self, key):
        return "?"


def trigger_text(call, style=PLAIN):
    if not isinstance(call, dict):
        return str(call)
    uri = call.get("uri")
    template = registry.facts(uri).get("describe") if isinstance(uri, str) and registry.lookup(uri) else None
    if template is None:
        return call_text(call, style=style)
    params = call.get("params") or {}
    return str(template).format_map(Shown({key: number(value) for key, value in params.items()}))


def sets_of(keys, style=PLAIN, every_set=("train", "valid", "test")):
    keys = keys or {}
    sets = keys.get("sets") or list(every_set)
    text = ", ".join(sets)
    every = keys.get("every")
    if every:
        text += style.dim(f"  every {number(every, style)}")
    return text


def compares_of(keys, style=PLAIN):
    keys = keys or {}
    output, target = keys.get("output"), keys.get("target")
    if output is None and target is None:
        return ""
    selector = target if isinstance(target, str) else (", ".join(target) if target else "")
    return f"{output or '—'} {ARROW} {selector or '—'}"


def definition_table(label, definitions, keys_table, style, width, notes, sets):
    if not definitions:
        return []
    keys_table = keys_table or {}
    rows = []
    for name, call in definitions.items():
        keys = keys_table.get(name) or {}
        rows.append([name, call_text(unwrap(call), style=style), compares_of(keys, style), sets_of(keys, style, sets),
                     style.dim(notes.get(name, ""))])
    if any(row[2] for row in rows):
        headers = [label, "lego", "compares", "reported on", ""]
    else:
        rows = [[row[0], row[1], row[3], row[4]] for row in rows]
        headers = [label, "lego", "reported on", ""]
    return ["", *table(headers, rows, style, width=width)]


def effect_text(value, style=PLAIN):
    if relative_effect(value):
        key, amount = next(iter(value.items()))
        return f"×{amount}" if key == "times" else f"+{amount}"
    if isinstance(value, dict) and "uri" in value:
        return call_text(value, style=style)
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key}: {effect_text(item, style)}" for key, item in value.items()) + "}"
    return str(value)


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
    sets = prepared.sets
    lines.extend(definition_table("loss", losses, params.get("losses_keys"), style, width, notes, sets))
    lines.extend(definition_table("metric", group_of(document, "metrics"), params.get("metrics_keys"), style,
                                  width, {}, sets))
    lines.append("")
    checkpoint = document.get("checkpoint") or None
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
            sets = ", ".join(f"{key} := {effect_text(value, style)}" for key, value in (rule.get("set") or {}).items())
            notes = [f"after {rule['after']}"] if rule.get("after") else []
            if rule.get("sticky") is False:
                notes.append("every turn")
            rows.append([when, ARROW, rule.get("name"), sets, style.dim(", ".join(notes))])
        lines.extend(table(["", "", "", "", ""], rows, style, indent="    ", width=width)[2:])
    return lines
