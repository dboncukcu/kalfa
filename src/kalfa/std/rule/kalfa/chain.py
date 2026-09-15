import copy
import logging
from pathlib import Path

from kalfa.std.common.effects import apply_effects, effect_note, relative_effect
from kalfa.std.common.files import read_note
from kalfa.std.common.log import logger_for
from kalfa.std.common.runtime import resolve_entries


logger = logger_for("training.rule")


def effects_text(targets):
    return ", ".join(f"{key}={effect_note(value)}" for key, value in (targets or {}).items())


def absolute(effects):
    return {key: value for key, value in (effects or {}).items() if not relative_effect(value)}


def open_rules(rules):
    out = copy.deepcopy(rules or {})
    out["fired"] = []
    out["pending"] = {}
    out["ready"] = list(dict.fromkeys([*(out.get("sticky") or []), *(out.get("ever") or [])]))
    return out


def rule(rules, name, when, set, after=None, metrics=None, turn_index=None, sticky=True):
    out = copy.deepcopy(rules)
    stuck = out.setdefault("sticky", [])
    pending = out.setdefault("pending", {})
    if name in stuck:
        pending.update(absolute(set))
        return out
    if after is not None and after not in (out.get("ready") or []):
        logger.debug(f"{name} waits for {after}")
        return out
    states = out.setdefault("triggers", {})
    fired, state = when(metrics, turn_index, states.get(name, {}))
    states[name] = {} if fired and not sticky else state
    if fired:
        if sticky:
            stuck.append(name)
        ever = out.setdefault("ever", [])
        if name not in ever:
            ever.append(name)
        out.setdefault("fired", []).append(name)
        pending.update(set or {})
        logger.info(f"{name} fired" + (f": {effects_text(set)}" if set else ""))
    elif logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"{name} not fired")
    return out


def effects(rules, models, optimizers, losses, loader):
    found = dict((rules or {}).get("effects") or {})
    resolve_entries(losses, loader=loader)
    return {"effects": found, "losses": apply_effects(found, models, optimizers, losses), "models": models,
            "optimizers": optimizers}


def stop_asked(record):
    note = read_note(Path(record) / "stop.json") if record is not None else None
    if note is None:
        return False
    if note.get("by"):
        logger.info(f"stopping after this turn: stop requested by {note['by']} at {note.get('at') or '?'}")
    else:
        logger.info("stopping after this turn: stop.json found in the record")
    return True


def stop(rules, triggers, metrics=None, record=None):
    out = copy.deepcopy(rules)
    states = list(out.get("stop") or [])
    fired = []
    for position, trigger in enumerate(triggers or []):
        state = states[position] if position < len(states) else {}
        hit, state = trigger(metrics, None, state)
        if position < len(states):
            states[position] = state
        else:
            states.append(state)
        fired.append(bool(hit))
    out["stop"] = states
    out["stop_fired"] = [position for position, hit in enumerate(fired) if hit]
    if out["stop_fired"]:
        logger.info("stopping after this turn: stop trigger "
                    + ", ".join(str(position) for position in out["stop_fired"]) + " fired")
    out["effects"] = dict(out.pop("pending", {}))
    out.pop("ready", None)
    return {"rules": out, "stop": any(fired) or stop_asked(record)}
