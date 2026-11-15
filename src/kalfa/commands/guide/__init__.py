from kalfa.cli import record_configs
from kalfa.commands import legos
from kalfa.commands.guide.about import ABOUT, COLORS, RUN, RUN_KEYS, UNKNOWN_COLOR, about_of
from kalfa.commands.guide.sections import (common_notes, data_notes, model_notes, param_notes, run_notes)
from kalfa.commands.guide.seen import Seen
from kalfa.commands.guide.show import head_renderable, renderable, section_renderable, text
from kalfa.commands.guide.training import (generate_notes, loss_notes, metric_notes, optimizer_notes,
                                           plot_notes, rule_notes, sweep_notes, train_notes)
from kalfa.commands.guide.words import unique, yaml_text
from kalfa.core import aliases, spec
from kalfa.core.check import check


__all__ = ["document", "head_renderable", "renderable", "section_renderable", "text"]


def no_notes(seen, key, value, keys, path):
    return []


NOTES = {"run": run_notes, "params": param_notes, "data": data_notes, "models": model_notes, "losses": loss_notes,
         "metrics": metric_notes, "optimizers": optimizer_notes, "train": train_notes, "rules": rule_notes,
         "plots": plot_notes, "generate": generate_notes, "sweep": sweep_notes}


def entry_of(seen, section, key, value, keys, path):
    try:
        notes = NOTES.get(section, no_notes)(seen, key, value, keys, path)
        notes += common_notes(seen, section, value, path, keys)
    except Exception as exc:
        notes = [f"kalfa could not explain this entry ({type(exc).__name__}: {exc})"]
    return {"path": path, "label": str(key), "yaml": yaml_text(key, value),
            "notes": unique(note for note in notes if note), "problems": []}


def pieces(seen, name, keys):
    if name == RUN:
        return [(key, seen.raw[key], (str(key),), str(key)) for key in keys]
    value = seen.raw[keys[0]]
    if name in ABOUT and isinstance(value, dict) and value:
        return [(inner, item, (name, inner), f"{name}.{inner}") for inner, item in value.items()]
    return [(keys[0], value, (name,), name)]


def sections_of(seen):
    order = {}
    for key in seen.raw:
        order.setdefault(RUN if str(key) in RUN_KEYS else str(key), []).append(key)
    return [{"name": name, "color": COLORS.get(name, UNKNOWN_COLOR), "about": about_of(name),
             "entries": [entry_of(seen, name, *piece) for piece in pieces(seen, name, keys)]}
            for name, keys in order.items()]


def problem_of(problem):
    return {"level": problem.severity, "code": problem.code, "path": problem.path, "message": problem.message,
            "hint": problem.hint}


def attach(sections, problems):
    entries = [entry for section in sections for entry in section["entries"]]
    for problem in problems:
        path = problem["path"]
        if not path:
            continue
        found = [entry for entry in entries
                 if path == entry["path"] or path.startswith((entry["path"] + ".", entry["path"] + "["))]
        if found:
            max(found, key=lambda entry: len(entry["path"]))["problems"].append(problem)


def failure(code, exc):
    return {"level": "error", "code": code, "path": None, "message": f"{type(exc).__name__}: {exc}", "hint": None}


def checked_alone(paths, sets, params):
    saved = list(aliases.SEARCH_DIRS)
    aliases.SEARCH_DIRS.clear()
    try:
        files = record_configs(paths)
        return files, check(paths=files, sets=spec.listed(sets), params=spec.listed(params))
    finally:
        aliases.SEARCH_DIRS[:] = saved


def document(paths, sets=(), params=()):
    paths = [str(path) for path in spec.listed(paths)]
    doc = {"title": " + ".join(paths) or "config", "ok": False, "problems": [], "sections": []}
    try:
        legos.load_registrars()
        files, checked = checked_alone(paths, sets, params)
    except Exception as exc:
        doc["problems"].append(failure("check_failed", exc))
        return doc
    doc["problems"] = [problem_of(problem) for problem in checked.problems]
    try:
        doc["sections"] = sections_of(Seen(checked, files))
    except Exception as exc:
        doc["problems"].append(failure("guide_failed", exc))
    attach(doc["sections"], doc["problems"])
    doc["ok"] = not any(problem["level"] == "error" for problem in doc["problems"])
    return doc
