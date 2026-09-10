from kalfa.registration import lego
from kalfa.std.common.history import History


@lego("/lego/kalfa/history", returns=None,
      bus=["monitor", "metrics", "turn_index", "counters_next", "optimizers_next", "rules_next", "record"],
      description="Append the turn's line to history.jsonl, its duration as seconds, and hand it to the monitor")
def history(monitor=None, metrics=None, turn_index=None, counters_next=None, optimizers_next=None, rules_next=None,
            record=None):
    line = History.line(metrics, counters_next, optimizers_next, rules_next,
                        monitor.elapsed() if monitor is not None else None)
    if record is not None:
        History.append(record, line)
    if monitor is not None:
        monitor.turn(line)
    return None
