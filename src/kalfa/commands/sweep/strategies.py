import copy
import difflib
import itertools
import json

import numpy

from kalfa.commands.sweep.space import draw, is_int, space_values, spread
from kalfa.core import aliases
from kalfa.core.errors import BuildError
from kalfa.records import reader
from kalfa.records.sweep import mode_of
from kalfa.std.common import positive_int


SAMPLERS = ("tpe", "random")
OPTUNA_MISSING = "optuna needs the optuna package (uv sync installs the dependencies of kalfa)"
STRATEGY_HINT = "grid, random, optuna or {use: <name or import path>, <param>: ...}"


def seed_value(seed):
    return None if seed is None else int(seed)


def resumed_seed(seed, count):
    return None if seed is None else int(numpy.random.SeedSequence([seed, count]).generate_state(1)[0])


class Grid:
    def __init__(self, steps=None):
        self.steps = None if steps is None else positive_int(steps, "grid steps")

    def axes(self, space):
        axes = {}
        for name, (kind, values) in space_values(space).items():
            if kind == "list":
                axes[name] = values
            elif self.steps is None:
                raise BuildError(f"sweep.space.{name}: the {kind} range needs steps on the grid, as in "
                                 f"strategy: {{use: grid, steps: 5}}, or a list of values")
            else:
                axes[name] = spread(kind, values, self.steps)
        return axes

    def points(self, space, seed=0):
        axes = self.axes(space)
        names = list(axes)
        return [dict(zip(names, copy.deepcopy(combination))) for combination in itertools.product(*axes.values())]


class Random:
    def __init__(self, trials, seed=None):
        self.trials = positive_int(trials, "random trials")
        self.seed = seed_value(seed)

    def points(self, space, seed=0):
        dimensions = space_values(space)
        generator = numpy.random.default_rng(self.seed if self.seed is not None else seed)
        found, seen = [], set()
        for _ in range(self.trials * 10):
            point = {name: draw(generator, kind, values) for name, (kind, values) in dimensions.items()}
            key = json.dumps(point, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            found.append(point)
            if len(found) == self.trials:
                break
        return found


def optuna_module():
    try:
        import optuna
    except ImportError as exc:
        raise BuildError(OPTUNA_MISSING) from exc
    return optuna


class Optuna:
    def __init__(self, trials, sampler="tpe", seed=None):
        self.trials = positive_int(trials, "optuna trials")
        if sampler not in SAMPLERS:
            raise BuildError(f"the optuna sampler is tpe or random, got {sampler!r}")
        self.sampler = sampler
        self.seed = seed_value(seed)
        self.study = None
        optuna_module()

    def start(self, objective, seed=None):
        optuna = optuna_module()
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        maker = optuna.samplers.TPESampler if self.sampler == "tpe" else optuna.samplers.RandomSampler
        direction = "maximize" if mode_of(objective) == "max" else "minimize"
        self.study = optuna.create_study(direction=direction, sampler=maker(seed=self.seed if seed is None else seed))
        return self.study

    def resume(self, space, objective, points):
        optuna = optuna_module()
        points = list(points)
        study = self.start(objective, resumed_seed(self.seed, len(points)))
        dimensions = space_values(space)
        told = 0
        for values, value in points:
            trial = finished_trial(optuna, dimensions, values, value)
            if trial is not None:
                study.add_trial(trial)
                told += 1
        return told

    def ask(self, space, objective):
        dimensions = space_values(space)
        study = self.study if self.study is not None else self.start(objective)
        trial = study.ask()
        return trial, {name: suggest(trial, name, kind, values) for name, (kind, values) in dimensions.items()}

    def tell(self, trial, value):
        optuna = optuna_module()
        if value is None or not reader.is_finite(value):
            self.study.tell(trial, state=optuna.trial.TrialState.FAIL)
        else:
            self.study.tell(trial, float(value))


def is_plain(value):
    return value is None or isinstance(value, (bool, int, float, str))


def suggest(trial, name, kind, values):
    if kind == "list":
        if all(is_plain(value) for value in values):
            return trial.suggest_categorical(name, values)
        return copy.deepcopy(values[trial.suggest_categorical(name, list(range(len(values))))])
    low, high = values
    if kind == "int":
        return int(trial.suggest_int(name, low, high))
    return float(trial.suggest_float(name, low, high, log=kind == "log"))


def distribution_of(optuna, kind, values, value):
    found = optuna.distributions
    if kind == "list":
        if all(is_plain(choice) for choice in values):
            return (found.CategoricalDistribution(values), value) if value in values else None
        index = next((index for index, choice in enumerate(values) if choice == value), None)
        return None if index is None else (found.CategoricalDistribution(list(range(len(values)))), index)
    low, high = values
    if not reader.is_finite(value) or not low <= value <= high:
        return None
    if kind == "int":
        return (found.IntDistribution(low, high), int(value)) if is_int(value) else None
    return found.FloatDistribution(low, high, log=kind == "log"), float(value)


def finished_trial(optuna, dimensions, values, value):
    if not isinstance(values, dict):
        return None
    params, distributions = {}, {}
    for name, (kind, choices) in dimensions.items():
        found = distribution_of(optuna, kind, choices, values[name]) if name in values else None
        if found is None:
            return None
        distributions[name], params[name] = found
    try:
        if value is None or not reader.is_finite(value):
            return optuna.trial.create_trial(params=params, distributions=distributions,
                                             state=optuna.trial.TrialState.FAIL)
        return optuna.trial.create_trial(params=params, distributions=distributions, value=float(value))
    except ValueError:
        return None


STRATEGIES = {"grid": Grid, "random": Random, "optuna": Optuna}


def strategy_call(raw):
    if raw is None:
        return "grid", {}
    if isinstance(raw, str) and raw:
        return raw, {}
    if isinstance(raw, dict) and isinstance(raw.get("use"), str) and raw["use"]:
        return raw["use"], {str(key): value for key, value in raw.items() if key != "use"}
    raise BuildError(f"sweep.strategy is {STRATEGY_HINT}, got {raw!r}")


def strategy_names():
    return [*STRATEGIES, *(entry.name for entry in aliases.catalog(slot="strategy") if entry.name not in STRATEGIES)]


def strategy_target(use):
    if use in STRATEGIES:
        return STRATEGIES[use]
    entry = aliases.entry_of(use)
    if entry is not None:
        if entry.slot not in (None, "strategy"):
            raise BuildError(f"{use!r} is a {entry.slot}, not a sweep strategy; the strategies are "
                             + ", ".join(strategy_names()))
        return aliases.import_target(entry.target)
    if aliases.is_import_path(use):
        return aliases.import_target(use)
    close = difflib.get_close_matches(use, strategy_names(), n=3, cutoff=0.6)
    hint = f"; did you mean {', '.join(close)}" if close else ""
    raise BuildError(f"{use!r} is not a sweep strategy; the strategies are {', '.join(strategy_names())} or an "
                     f"import path such as pkg.module.Class{hint}")


def strategy_of(raw):
    use, options = strategy_call(raw)
    target = strategy_target(use)
    try:
        strategy = target(**options)
    except BuildError:
        raise
    except Exception as exc:
        raise BuildError(f"sweep.strategy {use}: {type(exc).__name__}: {exc}") from exc
    if callable(getattr(strategy, "points", None)):
        return strategy
    if not (callable(getattr(strategy, "ask", None)) and callable(getattr(strategy, "tell", None))):
        raise BuildError(f"sweep.strategy {use}: a strategy has points(space, seed), or ask(space, objective) and "
                         f"tell(trial, value)")
    positive_int(getattr(strategy, "trials", None), f"sweep.strategy {use}: trials")
    return strategy


def strategy_note(raw):
    if isinstance(raw, dict):
        return copy.deepcopy(raw)
    return {"use": raw if isinstance(raw, str) else "grid"}
