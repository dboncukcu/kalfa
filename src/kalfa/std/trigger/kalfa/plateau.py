from kalfa.registration import lego
from kalfa.std.trigger.base import monitored


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
    current = monitored(metrics, monitor)
    if current is None:
        return False, state
    if improved(current, state.get("best"), mode, float(min_delta)):
        state["best"] = current
        state["wait"] = 0
    else:
        state["wait"] = int(state.get("wait", 0)) + 1
    return state["wait"] >= int(patience), state
