from kalfa.registration import lego
from kalfa.std.trigger.base import monitored


@lego("/trigger/kalfa/metric_above", partial=True, alias="metric_above",
      description="Fires when the monitored value rises above value; a missing value is not seen")
def metric_above(metrics, turn_index, state, monitor, value):
    current = monitored(metrics, monitor)
    if current is None:
        return False, dict(state or {})
    return current > float(value), dict(state or {})
