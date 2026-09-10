import json
import logging
import math
import time
from pathlib import Path

from kalfa.registration import lego
from kalfa.std.common.log import logger_for, number, turn_started


logger_turn = logger_for("training.turn")


def history_line(metrics, counters, optimizers, rules):
    line = {"turn": int((counters or {}).get("turn", 0)), "global_step": int((counters or {}).get("global_step", 0))}
    line.update(metrics or {})
    for name, optimizer in (optimizers or {}).items():
        lr = getattr(optimizer, "lr", None)
        if callable(lr):
            line[f"lr/{name}"] = lr()
    line["rules"] = list((rules or {}).get("fired") or [])
    return line


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
    line = history_line(metrics, counters_next, optimizers_next, rules_next)
    if record is not None:
        target = Path(record)
        target.mkdir(parents=True, exist_ok=True)
        with (target / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(line, default=float) + "\n")
    if logger_turn.isEnabledFor(logging.INFO):
        started = turn_started[0]
        logger_turn.info(turn_line(line, None if started is None else time.perf_counter() - started))
    if progress is not None:
        progress.update(line)
    return None
