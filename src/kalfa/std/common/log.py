import logging
import math
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

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
        if key in ("turn", "global_step", "rules", "seconds") or not isinstance(value, (int, float)):
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


def step_line(line):
    parts = [f"step {line.get('step')}"]
    for key, value in line.items():
        if key in ("step", "turn") or not isinstance(value, (int, float)):
            continue
        parts.append(f"{key} {number(float(value))}")
    return "  ".join(parts)


class Monitor:
    def __init__(self, level=None, progress=True, stream=None, log_every=None, tensorboard=False):
        self.level = level
        self.progress = progress
        self.stream = stream
        self.log_every = int(log_every) if log_every else None
        self.tensorboard = bool(tensorboard)
        self.handler = None
        self.bar = None
        self.inner = None
        self.total = None
        self.turn_started = None
        self.writer = None
        self.flow = logger_for("flow")
        self.turns = logger_for("training.turn")
        self.steps = logger_for("training.step")

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

    def open(self, record):
        if not self.tensorboard:
            return
        from kalfa.std.common.optional import load

        package = load("torch.utils.tensorboard", "the TensorBoard sink")
        if package is None:
            return
        self.writer = package.SummaryWriter(log_dir=str(Path(record) / "tensorboard"))

    def summary(self, params=None, last=None):
        if self.writer is None:
            return
        flat = {key: value for key, value in (params or {}).items() if isinstance(value, (int, float, str, bool))}
        values = {key: value for key, value in (last or {}).items()
                  if isinstance(value, (int, float)) and not isinstance(value, bool)}
        if flat and values:
            self.writer.add_hparams(flat, values)

    def finish(self):
        for bar in (self.inner, self.bar):
            if bar is not None:
                bar.close()
        self.inner = None
        self.bar = None
        self.total = None
        self.turn_started = None
        if self.writer is not None:
            self.writer.close()
            self.writer = None

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

    def elapsed(self):
        return None if self.turn_started is None else time.perf_counter() - self.turn_started

    def turn_begins(self, total):
        if self.inner is not None:
            self.inner.close()
            self.inner = None
        if self.progress != "steps":
            return
        from tqdm.auto import tqdm

        self.inner = tqdm(total=total, unit="step", dynamic_ncols=True, leave=False)

    def step(self, line):
        if self.log_every and int(line.get("step", 0)) % self.log_every == 0:
            self.steps.info(step_line(line))
        if self.writer is not None:
            for key, value in line.items():
                if key not in ("step", "turn") and isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, int(line.get("step", 0)))
        if self.inner is not None:
            losses = {key: value for key, value in line.items() if key.startswith("loss/")}
            self.inner.set_postfix({key: f"{value:.4g}" for key, value in list(losses.items())[:3]}, refresh=False)
            self.inner.update(1)

    def turn(self, line):
        if self.inner is not None:
            self.inner.close()
            self.inner = None
        if self.turns.isEnabledFor(logging.INFO):
            self.turns.info(turn_line(line, self.elapsed()))
        if self.writer is not None:
            for key, value in line.items():
                if key not in ("turn", "global_step", "rules") and isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, int(line.get("turn", 0)))
        if not self.progress:
            return
        from tqdm.auto import tqdm

        if self.bar is None:
            self.bar = tqdm(total=self.total, unit="turn", dynamic_ncols=True, leave=True)
        shown = {key: value for key, value in line.items()
                 if key != "seconds" and isinstance(value, float) and not math.isnan(value)}
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
