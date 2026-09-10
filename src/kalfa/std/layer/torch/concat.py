import torch
from torch import nn

from kalfa.registration import lego


class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, *values):
        return torch.cat(values, dim=self.dim)


@lego("/layer/torch/concat", alias="concat", description="Concatenate wires along a dimension")
def concat(dim=1):
    return Concat(dim)
