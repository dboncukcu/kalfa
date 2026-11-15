import os
import shlex
import subprocess
import sys
import tempfile

from rich.text import Text
from textual.message import Message
from textual.widgets import RichLog

from kalfa.core.errors import KalfaError


LOG_LINES = 5000


def words_of(text):
    try:
        return shlex.split(text or "")
    except ValueError as exc:
        raise KalfaError(f"cannot split {text!r}: {exc}") from None


def pairs_of(text, flag="--set"):
    words = words_of(text)
    bad = [word for word in words if "=" not in word or not word.partition("=")[0].strip()]
    if bad:
        raise KalfaError(f"{flag} takes NAME=VALUE pairs separated by spaces, got {' '.join(bad)}")
    return words


def cli_argv(command, *arguments, sets=(), params=()):
    argv = [sys.executable, "-m", "kalfa.cli", command, *(str(argument) for argument in arguments)]
    argv += [part for item in sets for part in ("--set", item)]
    return argv + [part for item in params for part in ("--param", item)]


def spawn(argv, label, detach=True, directory=None):
    handle, path = tempfile.mkstemp(prefix=f"kalfa-{label}-", suffix=".log", dir=directory)
    environment = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    with os.fdopen(handle, "wb") as output:
        process = subprocess.Popen([str(part) for part in argv], stdout=output, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, env=environment, start_new_session=detach)
    return process, path


def shown_line(raw):
    return raw.decode("utf-8", errors="replace").rstrip("\r").rsplit("\r", 1)[-1]


def follow(path, process, emit, stop, every=0.2, label="job"):
    offset, pending = 0, b""
    while not stop.is_set():
        finished = process.poll() is not None
        try:
            with open(path, "rb") as handle:
                handle.seek(offset)
                data = handle.read()
        except OSError:
            data = b""
        offset += len(data)
        *lines, pending = (pending + data).split(b"\n")
        for raw in lines:
            if emit(shown_line(raw)) is False:
                return None
        if finished and not data:
            if pending:
                emit(shown_line(pending))
            emit(f"{label} exited with {process.returncode}")
            return process.returncode
        if not data:
            stop.wait(every)
    return None


class JobOutput(Message):
    def __init__(self, text):
        super().__init__()
        self.text = text


class OutputLog(RichLog):
    def __init__(self, **options):
        super().__init__(max_lines=LOG_LINES, wrap=True, markup=False, highlight=False, **options)
        self.plain = []

    def say(self, text):
        text = str(text)
        self.plain.extend(text.splitlines() or [""])
        del self.plain[:-LOG_LINES]
        self.write(text)

    def on_job_output(self, message):
        self.say(message.text)

    def render(self):
        return Text("\n".join(self.plain))
