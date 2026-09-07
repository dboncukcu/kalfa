"""Logging: one history line per turn and the progress display."""

import json
import math
from pathlib import Path

from ..registration import lego


class Progress:
    """A tqdm bar over the turns; the total arrives from the loop's started event through ``expect``."""

    current = None

    def __init__(self):
        self.bar = None
        self.total = None
        Progress.current = self

    def expect(self, total):
        self.total = total
        if self.bar is not None:
            self.bar.total = total
            self.bar.refresh()

    def update(self, line):
        from tqdm import tqdm

        if self.bar is None:
            self.bar = tqdm(total=self.total, unit="turn", dynamic_ncols=True, leave=True)
        shown = {key: value for key, value in line.items()
                 if isinstance(value, float) and not math.isnan(value)}
        self.bar.set_postfix({key: f"{value:.4g}" for key, value in list(shown.items())[:4]}, refresh=False)
        self.bar.update(1)

    def close(self):
        if self.bar is not None:
            self.bar.close()
            self.bar = None


@lego("/lego/kalfa/progress", description="The progress display of a run")
def progress():
    return Progress()


def sink(event):
    """A tezgah event sink that hands the loop's turn total to the progress display."""
    if event.get("kind") == "started" and event.get("path", "").endswith("training.epochs") and "total" in event:
        if Progress.current is not None:
            Progress.current.expect(event["total"])


def history_line(metrics, counters, optimizers, rules):
    line = {"turn": int((counters or {}).get("turn", 0)), "global_step": int((counters or {}).get("global_step", 0))}
    line.update(metrics or {})
    for name, optimizer in (optimizers or {}).items():
        lr = getattr(optimizer, "lr", None)
        if callable(lr):
            line[f"lr/{name}"] = lr()
    line["rules"] = list((rules or {}).get("fired") or [])
    return line


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
    if progress is not None:
        progress.update(line)
    return None
