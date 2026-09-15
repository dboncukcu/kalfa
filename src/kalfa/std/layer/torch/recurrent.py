from torch import nn

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


class Gru(Recurrent):
    core_class = nn.GRU

    def __init__(self, hidden, layers=1, dropout=0.0, bidirectional=False):
        super().__init__(hidden, layers, dropout, bidirectional)


class Lstm(Recurrent):
    core_class = nn.LSTM

    def __init__(self, hidden, layers=1, dropout=0.0, bidirectional=False):
        super().__init__(hidden, layers, dropout, bidirectional)


class Rnn(Recurrent):
    core_class = nn.RNN

    def __init__(self, hidden, layers=1, dropout=0.0, bidirectional=False, nonlinearity="tanh"):
        super().__init__(hidden, layers, dropout, bidirectional, nonlinearity=nonlinearity)


def gru_cell(input_size, hidden, bias=True):
    return nn.GRUCell(int(input_size), int(hidden), bias=bool(bias))


def lstm_cell(input_size, hidden, bias=True):
    return nn.LSTMCell(int(input_size), int(hidden), bias=bool(bias))


def rnn_cell(input_size, hidden, bias=True, nonlinearity="tanh"):
    return nn.RNNCell(int(input_size), int(hidden), bias=bool(bias), nonlinearity=nonlinearity)
