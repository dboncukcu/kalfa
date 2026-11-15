from torch import nn


class MultiHead(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        sizes = [int(size) for size in hidden]
        layers = []
        for position, size in enumerate(sizes):
            layers.append(nn.LazyLinear(size) if position == 0 else nn.Linear(sizes[position - 1], size))
            layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleDict({name: nn.Linear(sizes[-1], int(width)) for name, width in heads.items()})

    def forward(self, x):
        shared = self.trunk(x)
        return {name: head(shared) for name, head in self.heads.items()}
