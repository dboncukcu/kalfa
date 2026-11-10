from importlib.metadata import PackageNotFoundError, version

from .core.aliases import lego, register, resolve
from .core.errors import BuildError, ConfigError, KalfaError, Problem, RunError
from .core.protocols import Action, ContextMetric, Logger, Objective, Step, Trigger


try:
    __version__ = version("kalfa")
except PackageNotFoundError:
    __version__ = "3.0.0.dev0"

from . import std  # noqa: E402
from .api import Monitor, Record, Rule, Run, Train, call, check, open, predict, run, summarize  # noqa: E402


__all__ = ["Action", "BuildError", "ConfigError", "ContextMetric", "KalfaError", "Logger", "Monitor", "Objective",
           "Problem", "Record", "Rule", "Run", "RunError", "Step", "Train", "Trigger", "__version__", "call", "check",
           "lego", "predict", "records", "register", "resolve", "run", "std", "summarize"]
