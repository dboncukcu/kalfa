import copy

from kalfa.registration import lego
from kalfa.std.common.log import logger_for


logger = logger_for("training.rule")


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
