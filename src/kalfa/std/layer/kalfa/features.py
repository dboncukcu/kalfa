from itertools import combinations, combinations_with_replacement

import torch
from torch import nn

from kalfa.std.common.deferred import later


class Select(nn.Module):
    def __init__(self, index, dim=1):
        super().__init__()
        self.dim = int(dim)
        self.register_buffer("index", torch.as_tensor([int(position) for position in index], dtype=torch.long))

    def forward(self, values):
        return torch.index_select(values, self.dim, self.index)


def select(index, dim=1):
    return later(Select, index=index, dim=dim)


class Polynomial(nn.Module):
    def __init__(self, degree=2, interaction_only=False, bias=False, keep=True):
        super().__init__()
        self.degree = int(degree)
        self.interaction_only = bool(interaction_only)
        self.bias = bool(bias)
        self.keep = bool(keep)
        if self.degree < 2:
            raise ValueError(f"degree must be 2 or more, got {degree!r}")
        self.index = None

    def index_tables(self, width, device):
        pick = combinations if self.interaction_only else combinations_with_replacement
        tables = []
        for order in range(2, self.degree + 1):
            rows = [list(entry) for entry in pick(range(width), order)]
            if rows:
                tables.append(torch.tensor(rows, dtype=torch.long, device=device))
        return tables

    def forward(self, values):
        flat = values.reshape(values.shape[0], -1)
        if self.index is None or not self.index or self.index[0].device != flat.device:
            self.index = self.index_tables(flat.shape[1], flat.device)
        parts = [flat] if self.keep else []
        parts.extend(flat[:, table].prod(dim=2) for table in self.index)
        if self.bias:
            parts.insert(0, torch.ones(flat.shape[0], 1, dtype=flat.dtype, device=flat.device))
        return torch.cat(parts, dim=1)


class L2Normalize(nn.Module):
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = float(eps)

    def forward(self, values):
        flat = values.reshape(values.shape[0], -1)
        return flat / flat.norm(dim=1, keepdim=True).clamp_min(self.eps)
