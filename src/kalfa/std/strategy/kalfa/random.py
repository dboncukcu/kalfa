from kalfa.registration import lego
from kalfa.std.strategy.base import Strategy, point_from_fractions


class RandomSearch(Strategy):
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


@lego("/strategy/kalfa/random", alias="random",
      description="count points drawn uniformly from the space with a seed; deterministic by id")
def random(count, seed=0):
    return RandomSearch(count, seed)
