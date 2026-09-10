from kalfa.registration import lego
from kalfa.std.strategy.base import Strategy, point_from_fractions
import numpy


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
