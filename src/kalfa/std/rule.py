"""Rules: the effects of the previous turn, the rule chain and the stop decision."""

import copy

from ..registration import lego


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
        return out
    states = out.setdefault("triggers", {})
    fired, state = when(metrics, turn_index, states.get(name, {}))
    states[name] = state
    if fired:
        sticky.append(name)
        out.setdefault("fired", []).append(name)
        pending.update(set or {})
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
    out["effects"] = dict(out.pop("pending", {}))
    out.pop("ready", None)
    return {"rules": out, "stop": any(fired)}
