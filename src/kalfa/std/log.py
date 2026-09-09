"""Logging: the console log of a run, one history line per turn and the progress display."""

import json
import logging
import math
import sys
import time
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from ..registration import lego
from ..style import style_for

ROOT = "kalfa"
LEVELS = {"info": logging.INFO, "debug": logging.DEBUG}
TAG_WIDTH = 14

logging.getLogger(ROOT).addHandler(logging.NullHandler())


def logger_for(name):
    return logging.getLogger(f"{ROOT}.{name}")


def level_of(name):
    return LEVELS.get(name) if name else None


def clock():
    return time.perf_counter()


def since(started):
    return f"{time.perf_counter() - started:.2f}s"


def number(value):
    return f"{value:.4g}" if isinstance(value, float) else str(value)


class Formatter(logging.Formatter):
    def __init__(self, style):
        super().__init__()
        self.style = style

    def format(self, record):
        stamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]
        name = record.name[len(ROOT) + 1:] if record.name.startswith(f"{ROOT}.") else record.name
        tag = getattr(record, "tag", None) or name
        line = f"{stamp}  {record.levelname.ljust(5)}  {tag.ljust(TAG_WIDTH)}  {record.getMessage()}"
        if record.levelno >= logging.ERROR:
            return self.style.red(line)
        if record.levelno >= logging.WARNING:
            return self.style.yellow(line)
        if record.levelno <= logging.DEBUG:
            return self.style.dim(line)
        return line


class Handler(logging.Handler):
    def __init__(self, stream=None):
        super().__init__()
        self.stream = stream

    def emit(self, record):
        try:
            stream = self.stream if self.stream is not None else sys.stderr
            text = self.format(record)
            if getattr(Progress.current, "bar", None) is not None:
                from tqdm.auto import tqdm

                tqdm.write(text, file=stream)
            else:
                stream.write(text + "\n")
                stream.flush()
        except Exception:
            self.handleError(record)


_handler = None


def start(level, stream=None):
    global _handler

    stop()
    if level is None:
        return None
    _handler = Handler(stream)
    _handler.setFormatter(Formatter(style_for(sys.stderr if stream is None else stream)))
    root = logging.getLogger(ROOT)
    root.setLevel(level)
    root.propagate = False
    root.addHandler(_handler)
    return _handler


def stop():
    global _handler

    root = logging.getLogger(ROOT)
    if _handler is not None:
        root.removeHandler(_handler)
        _handler = None
    root.setLevel(logging.NOTSET)
    root.propagate = True


def enabled():
    return _handler is not None


@contextmanager
def console(level, stream=None, progress=True):
    start(level, stream)
    Progress.enabled = progress
    try:
        yield
    finally:
        stop()
        Progress.enabled = True


def echo_warnings(collected):
    if not enabled():
        return

    def show(message, category, filename, lineno, file=None, line=None):
        collected.append(warnings.WarningMessage(message, category, filename, lineno, file, line))
        logger_for("warning").warning(str(message))

    warnings.showwarning = show


class Progress:
    """A tqdm bar over the turns; the total arrives from the loop's started event through ``expect``.

    The bar comes from ``tqdm.auto``, so a notebook draws the ipywidgets one and a terminal the plain one.
    ``enabled`` is the ``--no-progress`` switch: the bar is never created and tqdm never imported.
    """

    current = None
    enabled = True

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
        if not Progress.enabled:
            return
        from tqdm.auto import tqdm

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


logger_flow = logger_for("flow")
logger_turn = logger_for("training.turn")
_turn_started = [None]


def node_path(path):
    if path.endswith(".body"):
        path = path[:-len(".body")]
    return path.replace(".body.", ".")


def sink(event):
    """A tezgah event sink: the loop's turn total for the progress display, every node for the console log."""
    kind = event.get("kind")
    path = event.get("path") or ""
    if kind == "started" and path.endswith("training.epochs") and "total" in event:
        if Progress.current is not None:
            Progress.current.expect(event["total"])
    elif kind == "iter_started" and "epochs" in path:
        _turn_started[0] = time.perf_counter()
    if kind == "failed":
        logger_flow.error(f"failed: {event.get('error')}", extra={"tag": node_path(path)})
    elif not logger_flow.isEnabledFor(logging.DEBUG):
        return
    elif kind == "started":
        logger_flow.debug("started", extra={"tag": node_path(path)})
    elif kind == "finished":
        logger_flow.debug(f"finished ({float(event.get('ms') or 0) / 1000:.2f}s)", extra={"tag": node_path(path)})
    elif kind == "skipped":
        logger_flow.debug(f"skipped ({event.get('status')})", extra={"tag": node_path(path)})


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
        started = _turn_started[0]
        logger_turn.info(turn_line(line, None if started is None else time.perf_counter() - started))
    if progress is not None:
        progress.update(line)
    return None
