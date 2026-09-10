from torch import nn

from kalfa.registration import lego
from kalfa.std.common.deferred import LazyLayer


class Recurrent(LazyLayer):
    core_class = None

    def __init__(self, hidden, layers=1, dropout=0.0, bidirectional=False, **extra):
        super().__init__()
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.dropout = float(dropout)
        self.bidirectional = bool(bidirectional)
        self.extra = extra
        self.core = None

    def forward(self, value):
        if self.core is None:
            self.core = self.core_class(value.shape[-1], self.hidden, num_layers=self.layers, batch_first=True,
                                        dropout=self.dropout, bidirectional=self.bidirectional,
                                        **self.extra).to(device=value.device, dtype=value.dtype)
        out, _ = self.core(value)
        return out


@lego("/layer/torch/gru", alias="gru",
      description="GRU over (batch, steps, features) returning every step; the input width comes from the "
                  "first batch")
class Gru(Recurrent):
    core_class = nn.GRU

    def __init__(self, hidden, layers=1, dropout=0.0, bidirectional=False):
        super().__init__(hidden, layers, dropout, bidirectional)


@lego("/layer/torch/lstm", alias="lstm",
      description="LSTM over (batch, steps, features) returning every step; the input width comes from the "
                  "first batch")
class Lstm(Recurrent):
    core_class = nn.LSTM

    def __init__(self, hidden, layers=1, dropout=0.0, bidirectional=False):
        super().__init__(hidden, layers, dropout, bidirectional)


@lego("/layer/torch/rnn", alias="rnn",
      description="Elman RNN over (batch, steps, features) returning every step, nonlinearity tanh or relu; the "
                  "input width comes from the first batch")
class Rnn(Recurrent):
    core_class = nn.RNN

    def __init__(self, hidden, layers=1, dropout=0.0, bidirectional=False, nonlinearity="tanh"):
        super().__init__(hidden, layers, dropout, bidirectional, nonlinearity=nonlinearity)


@lego("/layer/torch/gru_cell", alias="gru_cell", description="torch.nn.GRUCell over one step; two inputs, x and h")
def gru_cell(input_size, hidden, bias=True):
    return nn.GRUCell(int(input_size), int(hidden), bias=bool(bias))


@lego("/layer/torch/lstm_cell", alias="lstm_cell",
      description="torch.nn.LSTMCell over one step; inputs x and (h, c), outputs (h, c)")
def lstm_cell(input_size, hidden, bias=True):
    return nn.LSTMCell(int(input_size), int(hidden), bias=bool(bias))


@lego("/layer/torch/rnn_cell", alias="rnn_cell", description="torch.nn.RNNCell over one step; two inputs, x and h")
def rnn_cell(input_size, hidden, bias=True, nonlinearity="tanh"):
    return nn.RNNCell(int(input_size), int(hidden), bias=bool(bias), nonlinearity=nonlinearity)
