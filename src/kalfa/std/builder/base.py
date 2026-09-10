from pathlib import Path

from torch import nn


class Model(nn.Module):
    inputs: list = ()
    outputs: list = ()
    initialized = True
    trainable = True


def weights_path(spec):
    which = spec.get("which")
    files = {"best": ("checkpoints", "best.pt"), "last": ("checkpoints", "last.pt"), "final": ("final", "state.pt")}
    if which not in files:
        raise ValueError(f"weights.which must be best, last or final, got {which!r}")
    return Path(spec["run"]).joinpath(*files[which])
