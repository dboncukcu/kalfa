"""Rules: the effects of the previous turn, the rule chain and the stop decision."""

import copy
import logging

from ..registration import lego
from .log import logger_for

logger = logger_for("training.rule")


def _brief(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return getattr(value, "__name__", type(value).__name__)


def _effects(targets):
    return ", ".join(f"{key}={_brief(value)}" for key, value in (targets or {}).items())


@lego("/rule/kalfa/effects", returns="effects",
            description="The effects the fired rules left for this turn")
def effects(rules):
    return dict((rules or {}).get("effects") or {})


@lego("/rule/kalfa/open", description="Open the rule chain of a turn")
def open(rules):
    out = copy.deepcopy(rules or {})
    out["fired"] = []
    out["pending"] = {}
    out["ready"] = list(out.get("sticky") or [])
    return out


@lego("/rule/kalfa/rule", returns="rules", bus=["metrics", "turn_index"],
            description="Evaluate one rule: skipped until its after rule fired in an earlier turn, sticky once "
                        "fired, later rules win the same key")
def rule(rules, name, when, set, after=None, metrics=None, turn_index=None):
    out = copy.deepcopy(rules)
    sticky = out.setdefault("sticky", [])
    pending = out.setdefault("pending", {})
    if name in sticky:
        pending.update(set or {})
        return out
    if after is not None and after not in (out.get("ready") or []):
        logger.debug(f"{name} waits for {after}")
        return out
    states = out.setdefault("triggers", {})
    fired, state = when(metrics, turn_index, states.get(name, {}))
    states[name] = state
    if fired:
        sticky.append(name)
        out.setdefault("fired", []).append(name)
        pending.update(set or {})
        logger.info(f"{name} fired" + (f": {_effects(set)}" if set else ""))
    elif logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"{name} not fired")
    return out


@lego("/rule/kalfa/stop", returns=["rules", "stop"], bus=["metrics"],
            description="Close the chain: the stop triggers are or'ed, their states kept under rules.stop")
def stop(rules, triggers, metrics=None):
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
    return {"rules": out, "stop": any(fired)}
