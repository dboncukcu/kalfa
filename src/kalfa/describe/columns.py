from cirak.registry import registry

from ..check import Checker
from ..std.common.runtime import expand_targets
from .document import block_params, field_plan, owners_of, target_fields
from .text import ARROW, table


def target_slots(prepared, probe=None):
    mapping = block_params(prepared, "after").get("targets") or {}
    fields = target_fields(prepared, probe)
    slots = {}
    for wire, selector in mapping.items():
        for position, name in enumerate(expand_targets(selector, fields)):
            slots[name] = (wire, position)
    return slots


def column_refs(prepared):
    found = {}
    try:
        for column, path in Checker(prepared.surface, registry, prepared.contract).column_refs():
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
                          "no column table; --measure builds it from the fitted plan")]
    fields, drop, _ = field_plan(prepared)
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
    rows.extend(extra_rows(probe, owners))
    lines = table(["column", "dtype", "field", "preprocessors", "role", "tensor"], rows, style, width=width)
    if probe is None:
        lines.append("  " + style.dim("the produced widths and the tensor slots need --measure"))
    return lines


def extra_rows(probe, owners=None):
    if probe is None or probe.prep is None:
        return []
    features = list(probe.prep.features)
    rows = []
    for item in probe.prep.fields:
        for extra in item.extras:
            slot = f"x[{features.index(extra)}]" if extra in features else "x"
            field = owners.get(item.name, item.name) if owners else item.name
            rows.append([extra, str(probe.prep.dtypes.get(extra, "?")), field, "a side output of the chain",
                         "feature", slot])
    return rows


def plan_columns(prepared, style, width, probe):
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
    rows.extend([[extra, dtype, chain, role, slot] for extra, dtype, chain, role, slot in
                 [(row[0], row[1], row[3], row[4], row[5]) for row in extra_rows(probe)]])
    return table(["column", "dtype", "preprocessors", "role", "tensor"], rows, style, width=width)
