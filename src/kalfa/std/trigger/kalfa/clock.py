import time


def after_turn(metrics, turn_index, state, at):
    seen = int((state or {}).get("seen", 0)) + 1
    return seen >= int(at), {"seen": seen}


def time_budget(metrics, turn_index, state, minutes):
    state = dict(state or {})
    now = time.time()
    started = state.setdefault("started", now)
    return (now - started) >= float(minutes) * 60.0, state
