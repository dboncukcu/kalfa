import logging
import math
import time

from kalfa.registration import lego
from kalfa.std.common.history import History
from kalfa.std.common.log import logger_for, number, turn_started


logger_turn = logger_for("training.turn")


def turn_line(line, elapsed=None):
    parts = [f"turn {line.get('turn')}"]
    for key, value in line.items():
        if key in ("turn", "global_step", "rules") or not isinstance(value, (int, float)):
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        parts.append(f"{key} {number(float(value))}")
    text = "  ".join(parts)
    fired = list(line.get("rules") or [])
    if fired:
        text += f"  rules: {', '.join(fired)}"
    if elapsed is not None:
        text += f"  ({elapsed:.1f}s)"
    return text


@lego("/lego/kalfa/history", returns=None,
      bus=["metrics", "turn_index", "counters_next", "optimizers_next", "rules_next", "record"],
      description="Append the turn's line to history.jsonl and advance the progress display")
def history(progress, metrics=None, turn_index=None, counters_next=None, optimizers_next=None, rules_next=None,
            record=None):
    line = History.line(metrics, counters_next, optimizers_next, rules_next)
    if record is not None:
        History.append(record, line)
    if logger_turn.isEnabledFor(logging.INFO):
        started = turn_started[0]
        logger_turn.info(turn_line(line, None if started is None else time.perf_counter() - started))
    if progress is not None:
        progress.update(line)
    return None
