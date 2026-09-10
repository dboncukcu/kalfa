from torch import nn

from kalfa.registration import lego


class LazyGRU(nn.Module):
    """A GRU whose input width is taken from the first batch; returns the output sequence (batch, steps, hidden)."""

    kalfa_lazy = True

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


@lego("/layer/torch/gru", alias="gru",
      description="GRU over (batch, steps, features) returning every step; the input width comes from the "
                  "first batch")
def gru(hidden, layers=1):
    return LazyGRU(hidden, layers)
