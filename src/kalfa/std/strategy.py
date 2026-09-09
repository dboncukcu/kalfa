"""Sweep strategies (kind: strategy): how the points of a search space are chosen.

A space maps a param name to a list of choices or a range ``{low, high, log, int, steps}``. Strategies deterministic by
id (grid, random, sobol) give the same point for the same id anywhere; a fed back strategy (optuna) proposes a point,
learns the objective and proposes the next, so it only runs in the local loop.
"""

import math

from ..registration import lego


RANGE_KEYS = ("low", "high", "log", "int", "steps")


class Choices:
    def __init__(self, values):
        self.values = list(values)


class Range:
    def __init__(self, low, high, log=False, integer=False, steps=None):
        self.low = float(low)
        self.high = float(high)
        self.log = bool(log)
        self.integer = bool(integer)
        self.steps = None if steps is None else int(steps)


def parse_space(mapping):
    """The space of a sweep section: name -> Choices or Range, validated."""
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("sweep.space maps param names to a list of choices or a range {low, high, log, int, steps}")
    space = {}
    for name, entry in mapping.items():
        if isinstance(entry, list):
            if not entry:
                raise ValueError(f"sweep.space.{name}: the list of choices is empty")
            space[name] = Choices(entry)
        elif isinstance(entry, dict):
            unknown = [key for key in entry if key not in RANGE_KEYS]
            if unknown or "low" not in entry or "high" not in entry:
                raise ValueError(f"sweep.space.{name}: a range is {{low, high, log, int, steps}}, got {sorted(entry)}")
            low, high = entry["low"], entry["high"]
            if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in (low, high)) \
                    or not low < high:
                raise ValueError(f"sweep.space.{name}: low must be a number below high")
            if entry.get("log") and low <= 0:
                raise ValueError(f"sweep.space.{name}: a log range needs low > 0")
            space[name] = Range(low, high, entry.get("log", False), entry.get("int", False), entry.get("steps"))
        else:
            raise ValueError(f"sweep.space.{name}: a list of choices or a range mapping")
    return space


def grid_values(name, entry):
    """The values a grid enumerates for one entry: the choices, or steps points of a range."""
    if isinstance(entry, Choices):
        return list(entry.values)
    if entry.steps is None or entry.steps < 2:
        raise ValueError(f"sweep.space.{name}: grid needs a list of choices or a range with steps >= 2")
    values = []
    for position in range(entry.steps):
        fraction = position / (entry.steps - 1)
        values.append(_at(entry, fraction))
    return values


def _at(entry, fraction):
    if entry.log:
        value = math.exp(math.log(entry.low) + fraction * (math.log(entry.high) - math.log(entry.low)))
    else:
        value = entry.low + fraction * (entry.high - entry.low)
    return int(round(value)) if entry.integer else float(value)


def sample(entry, fraction):
    """The value at a fraction of [0, 1): a choice by position, or a point of the range."""
    if isinstance(entry, Choices):
        return entry.values[min(int(fraction * len(entry.values)), len(entry.values) - 1)]
    return _at(entry, fraction)


def point_from_fractions(space, fractions):
    return {name: sample(entry, float(fraction)) for (name, entry), fraction in zip(space.items(), fractions)}


class Grid:
    """Every combination of the choices, the last name varying fastest."""

    deterministic = True

    def total(self, space):
        count = 1
        for name, entry in space.items():
            count *= len(grid_values(name, entry))
        return count

    def point(self, space, index):
        total = self.total(space)
        if not 0 <= index < total:
            raise ValueError(f"point {index} is outside the grid of {total} points")
        remaining = index
        picks = {}
        for name, entry in reversed(list(space.items())):
            values = grid_values(name, entry)
            remaining, position = divmod(remaining, len(values))
            picks[name] = values[position]
        return {name: picks[name] for name in space}


class RandomSearch:
    """count points drawn uniformly; point i comes from the generator seeded with (seed, i), so any id is
    reproducible without drawing the others."""

    deterministic = True

    def __init__(self, count, seed=0):
        self.count = int(count)
        self.seed = int(seed)

    def total(self, space):
        return self.count

    def point(self, space, index):
        import numpy

        if not 0 <= index < self.count:
            raise ValueError(f"point {index} is outside the {self.count} random points")
        generator = numpy.random.default_rng([self.seed, int(index)])
        return point_from_fractions(space, generator.random(len(space)))


class SobolSearch:
    """count points of a scrambled Sobol sequence; point i is the i-th draw of the seeded engine."""

    deterministic = True

    def __init__(self, count, seed=0, scramble=True):
        self.count = int(count)
        self.seed = int(seed)
        self.scramble = bool(scramble)

    def total(self, space):
        return self.count

    def point(self, space, index):
        import torch

        if not 0 <= index < self.count:
            raise ValueError(f"point {index} is outside the {self.count} sobol points")
        engine = torch.quasirandom.SobolEngine(dimension=len(space), scramble=self.scramble, seed=self.seed)
        draws = engine.draw(int(index) + 1)
        return point_from_fractions(space, draws[-1].tolist())


class OptunaSearch:
    """trials points proposed one by one by optuna's sampler from the objectives fed back; local loop only."""

    deterministic = False

    def __init__(self, trials, seed=0, sampler="tpe"):
        self.trials = int(trials)
        self.seed = int(seed)
        self.sampler = sampler
        self.study = None

    def total(self, space):
        return self.trials

    def _study(self, mode):
        if self.study is None:
            try:
                import optuna
            except ImportError as exc:
                raise ImportError("the optuna strategy needs optuna, which kalfa depends on; the environment "
                                  "is missing it, reinstall it with uv sync") from exc
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            if self.sampler == "random":
                sampler = optuna.samplers.RandomSampler(seed=self.seed)
            else:
                sampler = optuna.samplers.TPESampler(seed=self.seed)
            self.study = optuna.create_study(direction="minimize" if mode == "min" else "maximize", sampler=sampler)
        return self.study

    def ask(self, space, mode="min"):
        study = self._study(mode)
        trial = study.ask()
        point = {}
        for name, entry in space.items():
            if isinstance(entry, Choices):
                point[name] = trial.suggest_categorical(name, list(entry.values))
            elif entry.integer:
                point[name] = trial.suggest_int(name, int(entry.low), int(entry.high), log=entry.log)
            else:
                point[name] = trial.suggest_float(name, entry.low, entry.high, log=entry.log)
        return trial, point

    def tell(self, trial, value):
        import optuna

        if value is None or (isinstance(value, float) and math.isnan(value)):
            self.study.tell(trial, state=optuna.trial.TrialState.FAIL)
        else:
            self.study.tell(trial, float(value))


@lego("/strategy/kalfa/grid", alias="grid",
            description="Every combination of the space's choices (a range needs steps); deterministic by id")
def grid():
    return Grid()


@lego("/strategy/kalfa/random", alias="random",
            description="count points drawn uniformly from the space with a seed; deterministic by id")
def random(count, seed=0):
    return RandomSearch(count, seed)


@lego("/strategy/kalfa/sobol", alias="sobol",
            description="count points of a scrambled Sobol sequence with a seed; deterministic by id")
def sobol(count, seed=0, scramble=True):
    return SobolSearch(count, seed, scramble)


@lego("/strategy/kalfa/optuna", alias="optuna",
            description="trials points proposed by optuna (tpe or random sampler) from the objectives fed back; "
                        "local loop only, no --id")
def optuna(trials, seed=0, sampler="tpe"):
    return OptunaSearch(trials, seed, sampler)
