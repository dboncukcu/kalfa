import math

from kalfa.registration import lego


def monitored(metrics, monitor):
    value = (metrics or {}).get(monitor)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)


def improved(value, best, mode, min_delta):
    if best is None:
        return True
    if mode == "max":
        return value > best + min_delta
    return value < best - min_delta


@lego("/trigger/kalfa/metric_above", partial=True, alias="metric_above", describe="{monitor} > {value}",
      description="Fires when the monitored value rises above value; a missing value is not seen")
def metric_above(metrics, turn_index, state, monitor, value):
    current = monitored(metrics, monitor)
    if current is None:
        return False, dict(state or {})
    return current > float(value), dict(state or {})


@lego("/trigger/kalfa/metric_below", partial=True, alias="metric_below", describe="{monitor} < {value}",
      description="Fires when the monitored value drops below value; a missing value is not seen")
def metric_below(metrics, turn_index, state, monitor, value):
    current = monitored(metrics, monitor)
    if current is None:
        return False, dict(state or {})
    return current < float(value), dict(state or {})


@lego("/trigger/kalfa/plateau", partial=True, alias="plateau", describe="{monitor} plateau {patience}",
      description="Fires after patience turns without improvement of the monitored value; "
                  "turns without the value are not counted")
def plateau(metrics, turn_index, state, monitor, patience, mode="min", min_delta=0.0):
    state = dict(state or {})
    current = monitored(metrics, monitor)
    if current is None:
        return False, state
    if improved(current, state.get("best"), mode, float(min_delta)):
        state["best"] = current
        state["wait"] = 0
    else:
        state["wait"] = int(state.get("wait", 0)) + 1
    return state["wait"] >= int(patience), state
