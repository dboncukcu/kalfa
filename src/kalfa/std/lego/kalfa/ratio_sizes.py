from kalfa.registration import lego
from kalfa.std.split.base import cuts_of


@lego("/lego/kalfa/ratio_sizes",
      description="The set sizes a split by ratios produces from rows rows; without rows, which sets it produces")
def ratio_sizes(rows, ratios, seed=None, group=None):
    if rows is None:
        cuts_of(1, ratios)
        return {name: (None if ratio > 0 else 0) for name, ratio in zip(("train", "valid", "test"), ratios)}
    first, second = cuts_of(rows, ratios)
    return {"train": first, "valid": second - first, "test": rows - second}
