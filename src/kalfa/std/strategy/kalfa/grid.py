from kalfa.registration import lego
from kalfa.std.strategy.base import Strategy, grid_values


@lego("/strategy/kalfa/grid", alias="grid",
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
