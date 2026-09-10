from kalfa.registration import lego
from kalfa.std.trigger.base import monitored


@lego("/trigger/kalfa/metric_below", partial=True, alias="metric_below", describe="{monitor} < {value}",
      description="Fires when the monitored value drops below value; a missing value is not seen")
def metric_below(metrics, turn_index, state, monitor, value):
    current = monitored(metrics, monitor)
    if current is None:
        return False, dict(state or {})
    return current < float(value), dict(state or {})
