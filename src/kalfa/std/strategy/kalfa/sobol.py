from kalfa.registration import lego
from kalfa.std.strategy.base import Strategy, point_from_fractions


class SobolSearch(Strategy):
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


@lego("/strategy/kalfa/sobol", alias="sobol",
      description="count points of a scrambled Sobol sequence with a seed; deterministic by id")
def sobol(count, seed=0, scramble=True):
    return SobolSearch(count, seed, scramble)
