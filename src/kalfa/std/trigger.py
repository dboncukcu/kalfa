"""Triggers: partial legos with update(metrics, turn_index, state) -> (fired, state); used by rules and stop."""

import math
import time

from ..registration import lego


def _monitored(metrics, monitor):
    value = (metrics or {}).get(monitor)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)


@lego("/trigger/kalfa/after_turn", partial=True, alias=["after_turn", "after_epoch"],
            description="Fires once the given number of turns has ended, counted across resumes")
def after_turn(metrics, turn_index, state, at):
    seen = int((state or {}).get("seen", 0)) + 1
    return seen >= int(at), {"seen": seen}


@lego("/trigger/kalfa/metric_below", partial=True, alias="metric_below",
            description="Fires when the monitored value drops below value; a missing value is not seen")
def metric_below(metrics, turn_index, state, monitor, value):
    current = _monitored(metrics, monitor)
    if current is None:
        return False, dict(state or {})
    return current < float(value), dict(state or {})


@lego("/trigger/kalfa/metric_above", partial=True, alias="metric_above",
            description="Fires when the monitored value rises above value; a missing value is not seen")
def metric_above(metrics, turn_index, state, monitor, value):
    current = _monitored(metrics, monitor)
    if current is None:
        return False, dict(state or {})
    return current > float(value), dict(state or {})


def improved(value, best, mode, min_delta):
    if best is None:
        return True
    if mode == "max":
        return value > best + min_delta
    return value < best - min_delta


@lego("/trigger/kalfa/plateau", partial=True, alias="plateau",
            description="Fires after patience turns without improvement of the monitored value; "
                        "turns without the value are not counted")
def plateau(metrics, turn_index, state, monitor, patience, mode="min", min_delta=0.0):
    state = dict(state or {})
    current = _monitored(metrics, monitor)
    if current is None:
        return False, state
    if improved(current, state.get("best"), mode, float(min_delta)):
        state["best"] = current
        state["wait"] = 0
    else:
        state["wait"] = int(state.get("wait", 0)) + 1
    return state["wait"] >= int(patience), state


@lego("/trigger/kalfa/time_budget", partial=True, alias="time_budget",
            description="Fires once the given number of minutes has passed since the first turn it saw")
def time_budget(metrics, turn_index, state, minutes):
    state = dict(state or {})
    now = time.time()
    started = state.setdefault("started", now)
    return (now - started) >= float(minutes) * 60.0, state
