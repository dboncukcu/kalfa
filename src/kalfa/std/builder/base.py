from torch import nn


class Model(nn.Module):
    inputs = ()
    outputs = ()
    initialized = True
    kalfa_trainable = True


WEIGHT_FILES = {"best": ("checkpoints", "best.pt"), "last": ("checkpoints", "last.pt"),
                "final": ("final", "state.pt")}


def weights_path(spec):
    """The checkpoint file a weights spec {run, model, which} names."""
    from pathlib import Path

    which = spec.get("which")
    if which not in WEIGHT_FILES:
        raise ValueError(f"weights.which must be best, last or final, got {which!r}")
    return Path(spec["run"]).joinpath(*WEIGHT_FILES[which])
