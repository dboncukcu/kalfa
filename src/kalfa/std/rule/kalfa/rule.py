import copy
import logging

from kalfa.registration import lego
from kalfa.std.common.log import logger_for


logger = logger_for("training.rule")


def brief(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return getattr(value, "__name__", type(value).__name__)


def effects_text(targets):
    return ", ".join(f"{key}={brief(value)}" for key, value in (targets or {}).items())


def absolute(effects):
    return {key: value for key, value in (effects or {}).items() if not isinstance(value, dict)}


@lego("/rule/kalfa/rule", returns="rules", bus=["metrics", "turn_index"],
      description="Evaluate one rule: skipped until its after rule fired in an earlier turn; a sticky rule keeps "
                  "its effects once fired and is not asked again; with sticky false it is asked every turn, its "
                  "relative effects (times, plus) apply once per firing and its trigger starts over; later rules "
                  "win the same key")
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
