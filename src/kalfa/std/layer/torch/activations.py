from torch import nn


def relu():
    return nn.ReLU()


def relu6():
    return nn.ReLU6()


def leaky_relu(negative_slope=0.01):
    return nn.LeakyReLU(negative_slope)


def prelu(num_parameters=1, init=0.25):
    return nn.PReLU(int(num_parameters), float(init))


def rrelu(lower=0.125, upper=1.0 / 3.0):
    return nn.RReLU(float(lower), float(upper))


def elu(alpha=1.0):
    return nn.ELU(float(alpha))


def celu(alpha=1.0):
    return nn.CELU(float(alpha))


def selu():
    return nn.SELU()


def gelu(approximate="none"):
    return nn.GELU(approximate=approximate)


def silu():
    return nn.SiLU()


def mish():
    return nn.Mish()


def hardswish():
    return nn.Hardswish()


def hardsigmoid():
    return nn.Hardsigmoid()


def hardtanh(min_val=-1.0, max_val=1.0):
    return nn.Hardtanh(float(min_val), float(max_val))


def hardshrink(lambd=0.5):
    return nn.Hardshrink(float(lambd))


def softshrink(lambd=0.5):
    return nn.Softshrink(float(lambd))


def tanhshrink():
    return nn.Tanhshrink()


def softsign():
    return nn.Softsign()


def softplus(beta=1.0, threshold=20.0):
    return nn.Softplus(float(beta), float(threshold))


def sigmoid():
    return nn.Sigmoid()


def log_sigmoid():
    return nn.LogSigmoid()


def tanh():
    return nn.Tanh()


def threshold(threshold, value):
    return nn.Threshold(float(threshold), float(value))


def glu(dim=-1):
    return nn.GLU(int(dim))


def softmax(dim=-1):
    return nn.Softmax(int(dim))


def softmin(dim=-1):
    return nn.Softmin(int(dim))


def log_softmax(dim=-1):
    return nn.LogSoftmax(int(dim))


def softmax2d():
    return nn.Softmax2d()
