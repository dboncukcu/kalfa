import logging
import math
import sys
import time
import warnings
from datetime import datetime

from kalfa.style import style_for


ROOT = "kalfa"
TAG_WIDTH = 14
logging.getLogger(ROOT).addHandler(logging.NullHandler())


def logger_for(name):
    return logging.getLogger(f"{ROOT}.{name}")


def level_of(name):
    if not name:
        return None
    level = logging.getLevelName(name.upper())
    return level if isinstance(level, int) else None


def clock():
    return time.perf_counter()


def since(started):
    return f"{time.perf_counter() - started:.2f}s"


def number(value):
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def node_path(path):
    if path.endswith(".body"):
        path = path[:-len(".body")]
    return path.replace(".body.", ".")


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
    def __init__(self, monitor, stream=None):
        super().__init__()
        self.monitor = monitor
        self.stream = stream

    def emit(self, record):
        try:
            stream = self.stream if self.stream is not None else sys.stderr
            text = self.format(record)
            if self.monitor.bar is not None:
                from tqdm.auto import tqdm

                tqdm.write(text, file=stream)
            else:
                stream.write(text + "\n")
                stream.flush()
        except Exception:
            self.handleError(record)


class Monitor:
    def __init__(self, level=None, progress=True, stream=None):
        self.level = level
        self.progress = progress
        self.stream = stream
        self.handler = None
        self.bar = None
        self.total = None
        self.turn_started = None
        self.flow = logger_for("flow")
        self.turns = logger_for("training.turn")

    def __enter__(self):
        return self.start()

    def __exit__(self, *error):
        self.close()

    def start(self):
        self.stop()
        if self.level is None:
            return self
        self.handler = Handler(self, self.stream)
        self.handler.setFormatter(Formatter(style_for(sys.stderr if self.stream is None else self.stream)))
        root = logging.getLogger(ROOT)
        root.setLevel(self.level)
        root.propagate = False
        root.addHandler(self.handler)
        return self

    def stop(self):
        root = logging.getLogger(ROOT)
        if self.handler is not None:
            root.removeHandler(self.handler)
            self.handler = None
        root.setLevel(logging.NOTSET)
        root.propagate = True

    def finish(self):
        if self.bar is not None:
            self.bar.close()
            self.bar = None
        self.total = None
        self.turn_started = None

    def close(self):
        self.finish()
        self.stop()

    @property
    def logging(self):
        return self.handler is not None

    def echo_warnings(self, collected):
        if not self.logging:
            return

        def show(message, category, filename, lineno, file=None, line=None):
            collected.append(warnings.WarningMessage(message, category, filename, lineno, file, line))
            logger_for("warning").warning(str(message))

        warnings.showwarning = show

    def expect(self, total):
        self.total = total
        if self.bar is not None:
            self.bar.total = total
            self.bar.refresh()

    def turn(self, line):
        if self.turns.isEnabledFor(logging.INFO):
            elapsed = None if self.turn_started is None else time.perf_counter() - self.turn_started
            self.turns.info(turn_line(line, elapsed))
        if not self.progress:
            return
        from tqdm.auto import tqdm

        if self.bar is None:
            self.bar = tqdm(total=self.total, unit="turn", dynamic_ncols=True, leave=True)
        shown = {key: value for key, value in line.items() if isinstance(value, float) and not math.isnan(value)}
        self.bar.set_postfix({key: f"{value:.4g}" for key, value in list(shown.items())[:4]}, refresh=False)
        self.bar.update(1)

    def sink(self, event):
        kind = event.get("kind")
        path = event.get("path") or ""
        if kind == "started" and path.endswith("training.epochs") and "total" in event:
            self.expect(event["total"])
        elif kind == "iter_started" and "epochs" in path:
            self.turn_started = time.perf_counter()
        if kind == "failed":
            self.flow.error(f"failed: {event.get('error')}", extra={"tag": node_path(path)})
        elif not self.flow.isEnabledFor(logging.DEBUG):
            return
        elif kind == "started":
            self.flow.debug("started", extra={"tag": node_path(path)})
        elif kind == "finished":
            self.flow.debug(f"finished ({float(event.get('ms') or 0) / 1000:.2f}s)", extra={"tag": node_path(path)})
        elif kind == "skipped":
            self.flow.debug(f"skipped ({event.get('status')})", extra={"tag": node_path(path)})
