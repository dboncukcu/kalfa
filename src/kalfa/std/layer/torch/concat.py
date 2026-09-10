import torch
from torch import nn

from kalfa.registration import lego


@lego("/layer/torch/concat", alias="concat", description="Concatenate wires along a dimension")
class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, *values):
        return torch.cat(values, dim=self.dim)
