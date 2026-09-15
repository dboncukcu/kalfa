from kalfa.std.common.effects import effect_note, effective_loss
from kalfa.std.common.history import History


def minimized(effects, optimizers):
    return {name: effective_loss(name, effects or {}, optimizers) for name, optimizer in (optimizers or {}).items()
            if optimizer.loss is not None or f"{name}.loss" in (effects or {})}


def noted(effects):
    return {target: effect_note(value) for target, value in (effects or {}).items()
            if target != "loss" and not target.endswith(".loss")}


def history(monitor=None, metrics=None, turn_index=None, counters_next=None, optimizers_next=None, rules_next=None,
            effects=None, record=None):
    line = History.line(metrics, counters_next, optimizers_next, rules_next,
                        monitor.elapsed() if monitor is not None else None, minimized(effects, optimizers_next),
                        noted(effects))
    if record is not None:
        History.append(record, line)
    if monitor is not None:
        monitor.turn(line)
    return None
