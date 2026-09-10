from torch import nn

from kalfa.registration import lego
from kalfa.std.layer.base import LazyLayer


@lego("/layer/torch/gru", alias="gru",
      description="GRU over (batch, steps, features) returning every step; the input width comes from the "
                  "first batch")
class Gru(LazyLayer):
    def __init__(self, hidden, layers=1):
        super().__init__()
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.core = None

    def forward(self, value):
        if self.core is None:
            self.core = nn.GRU(value.shape[-1], self.hidden, num_layers=self.layers, batch_first=True).to(
                device=value.device, dtype=value.dtype)
        out, _ = self.core(value)
        return out
