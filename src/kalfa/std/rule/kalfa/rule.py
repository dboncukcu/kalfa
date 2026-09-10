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
        logger.info(f"{name} fired" + (f": {effects_text(set)}" if set else ""))
    elif logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"{name} not fired")
    return out
