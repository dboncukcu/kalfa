import logging
import math
import sys
import time
import warnings
from contextlib import contextmanager
from datetime import datetime

from kalfa.style import style_for


ROOT = "kalfa"
TAG_WIDTH = 14
logging.getLogger(ROOT).addHandler(logging.NullHandler())


def logger_for(name):
    return logging.getLogger(f"{ROOT}.{name}")


logger_flow = logger_for("flow")


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
            if Progress.current is not None and Progress.current.bar is not None:
                from tqdm.auto import tqdm

                tqdm.write(text, file=stream)
            else:
                stream.write(text + "\n")
                stream.flush()
        except Exception:
            self.handleError(record)


handler = None


def start(level, stream=None):
    global handler

    stop()
    if level is None:
        return None
    handler = Handler(stream)
    handler.setFormatter(Formatter(style_for(sys.stderr if stream is None else stream)))
    root = logging.getLogger(ROOT)
    root.setLevel(level)
    root.propagate = False
    root.addHandler(handler)
    return handler


def stop():
    global handler

    root = logging.getLogger(ROOT)
    if handler is not None:
        root.removeHandler(handler)
        handler = None
    root.setLevel(logging.NOTSET)
    root.propagate = True


def enabled():
    return handler is not None


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


turn_started = [None]


def node_path(path):
    if path.endswith(".body"):
        path = path[:-len(".body")]
    return path.replace(".body.", ".")


def sink(event):
    kind = event.get("kind")
    path = event.get("path") or ""
    if kind == "started" and path.endswith("training.epochs") and "total" in event:
        if Progress.current is not None:
            Progress.current.expect(event["total"])
    elif kind == "iter_started" and "epochs" in path:
        turn_started[0] = time.perf_counter()
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
