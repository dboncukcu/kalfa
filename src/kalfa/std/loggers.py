import sys

from kalfa.core.aliases import register
from kalfa.core.errors import BuildError
from kalfa.core.protocols import Logger
from kalfa.std.common import numeric, positive_int


MISSING = "tensorboard logger needs the tensorboard package (uv sync installs the dependencies of kalfa)"
COUNTERS = ("turn", "update", "seconds")
PER_UPDATE = ("loss/", "lr/", "grad_norm/")


def summary_writer():
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise BuildError(MISSING) from exc
    return SummaryWriter


def per_update(key):
    return key.startswith(PER_UPDATE) and key.count("/") == 1


def series_of(line):
    return [key for key, value in line.items() if key not in COUNTERS and numeric(value)]


class TensorBoard(Logger):
    def __init__(self, dir=None, updates=True, flush_every=1):
        self.dir = dir
        self.updates = bool(updates)
        self.flush_every = positive_int(flush_every, "flush_every")
        self.writer = None
        self.writer_class = None
        self.turns = 0

    def bind(self, ctx):
        self.writer_class = summary_writer()

    def directory(self, ctx):
        if self.dir is not None:
            return str(self.dir)
        record = getattr(ctx, "record", None)
        if record is None:
            where = f"{self.path}: " if self.path else ""
            raise BuildError(f"{where}the tensorboard logger needs a dir when the run writes no record")
        return str(record.path("tensorboard"))

    def open(self, ctx):
        if self.writer is None:
            if self.writer_class is None:
                self.writer_class = summary_writer()
            self.writer = self.writer_class(log_dir=self.directory(ctx))
        return self.writer

    def turn(self, ctx, line):
        writer = self.open(ctx)
        step = int(line["turn"])
        for key in series_of(line):
            if self.updates and per_update(key):
                continue
            writer.add_scalar(key, float(line[key]), step)
        self.turns += 1
        if self.turns % self.flush_every == 0:
            writer.flush()

    def update(self, ctx, line):
        if not self.updates:
            return
        writer = self.open(ctx)
        step = int(line["update"])
        for key in series_of(line):
            if per_update(key):
                writer.add_scalar(key, float(line[key]), step)

    def close(self, ctx):
        if self.writer is None:
            return
        writer, self.writer = self.writer, None
        writer.flush()
        writer.close()


class Echo(Logger):
    def __init__(self, series=None, stream=None):
        self.series = None if series is None else [series] if isinstance(series, str) else list(series)
        self.stream = stream

    def turn(self, ctx, line):
        keys = series_of(line) if self.series is None else self.series
        pairs = [f"{key} {line[key]:.4g}" for key in keys if numeric(line.get(key))]
        stream = sys.stderr if self.stream is None else self.stream
        stream.write("  ".join([f"turn {line['turn']}", *pairs]) + "\n")
        flush = getattr(stream, "flush", None)
        if callable(flush):
            flush()


register("tensorboard", "kalfa.std.loggers.TensorBoard", slot="logger",
         doc="writes every history series per turn and the update losses to <record>/tensorboard for TensorBoard; "
             "needs the tensorboard package (uv sync installs the dependencies of kalfa)")
register("echo", "kalfa.std.loggers.Echo", slot="logger",
         doc="prints the named series (or all) per turn to stderr; a logger to copy when writing your own")
