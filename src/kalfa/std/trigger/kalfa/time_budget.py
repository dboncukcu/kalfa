import time

from kalfa.registration import lego


@lego("/trigger/kalfa/time_budget", partial=True, alias="time_budget", describe="after {minutes} minutes",
      description="Fires once the given number of minutes has passed since the first turn it saw")
def time_budget(metrics, turn_index, state, minutes):
    state = dict(state or {})
    now = time.time()
    started = state.setdefault("started", now)
    return (now - started) >= float(minutes) * 60.0, state
