from kalfa.registration import lego
from kalfa.std.common.history import History
from kalfa.std.turn.base import effective_loss


def minimized(effects, optimizers):
    return {name: effective_loss(name, effects or {}, optimizers) for name, optimizer in (optimizers or {}).items()
            if optimizer.loss is not None or f"{name}.loss" in (effects or {})}


@lego("/lego/kalfa/history", returns=None,
      bus=["monitor", "metrics", "turn_index", "counters_next", "optimizers_next", "rules_next", "effects", "record"],
      description="Append the turn's line to history.jsonl: the metrics, the learning rate and the loss every "
                  "optimizer minimized this turn (as the rules set it), the duration as seconds, the rules that "
                  "fired; and hand it to the monitor")
def history(monitor=None, metrics=None, turn_index=None, counters_next=None, optimizers_next=None, rules_next=None,
            effects=None, record=None):
    line = History.line(metrics, counters_next, optimizers_next, rules_next,
                        monitor.elapsed() if monitor is not None else None, minimized(effects, optimizers_next))
    if record is not None:
        History.append(record, line)
    if monitor is not None:
        monitor.turn(line)
    return None
