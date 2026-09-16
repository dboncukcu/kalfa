import torch
from torch import nn


def multiplier_starts(names, init):
    if isinstance(names, dict):
        entries = {str(name): spec for name, spec in names.items()}
    else:
        entries = {str(name): None for name in (names or [])}
    if not entries:
        raise ValueError("multipliers needs names: the constraints mapping of the mdmm loss, or a list of names")
    starts = []
    for spec in entries.values():
        start = spec.get("lmbda_init", init) if isinstance(spec, dict) else init
        starts.append(float(start))
    return list(entries), starts


class Multipliers(nn.Module):
    def __init__(self, names, init=0.0):
        super().__init__()
        self.names, starts = multiplier_starts(names, init)
        self.lmbda = nn.Parameter(torch.tensor(starts, dtype=torch.float32))

    def forward(self, *values):
        return self.lmbda
