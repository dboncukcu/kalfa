import numpy
import torch

from kalfa.registration import lego
from kalfa.std.strategy.base import Strategy, grid_values, point_from_fractions


@lego("/strategy/kalfa/grid", alias="grid", enumerates=True,
      description="Every combination of the space's choices (a range needs steps); deterministic by id")
class Grid(Strategy):
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


@lego("/strategy/kalfa/random", alias="random",
      description="count points drawn uniformly from the space with a seed; deterministic by id")
class RandomSearch(Strategy):
    deterministic = True

    def __init__(self, count, seed=0):
        self.count = int(count)
        self.seed = int(seed)

    def total(self, space):
        return self.count

    def point(self, space, index):
        if not 0 <= index < self.count:
            raise ValueError(f"point {index} is outside the {self.count} random points")
        generator = numpy.random.default_rng([self.seed, int(index)])
        return point_from_fractions(space, generator.random(len(space)))


@lego("/strategy/kalfa/sobol", alias="sobol",
      description="count points of a scrambled Sobol sequence with a seed; deterministic by id")
class SobolSearch(Strategy):
    deterministic = True

    def __init__(self, count, seed=0, scramble=True):
        self.count = int(count)
        self.seed = int(seed)
        self.scramble = bool(scramble)

    def total(self, space):
        return self.count

    def point(self, space, index):
        if not 0 <= index < self.count:
            raise ValueError(f"point {index} is outside the {self.count} sobol points")
        engine = torch.quasirandom.SobolEngine(dimension=len(space), scramble=self.scramble, seed=self.seed)
        draws = engine.draw(int(index) + 1)
        return point_from_fractions(space, draws[-1].tolist())
