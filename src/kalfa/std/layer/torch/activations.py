from torch import nn

from kalfa.registration import lego


@lego("/layer/torch/relu", alias="relu", description="torch.nn.ReLU")
def relu():
    return nn.ReLU()


@lego("/layer/torch/relu6", alias="relu6", description="torch.nn.ReLU6, ReLU clipped at 6")
def relu6():
    return nn.ReLU6()


@lego("/layer/torch/leaky_relu", alias="leaky_relu", description="torch.nn.LeakyReLU with negative_slope")
def leaky_relu(negative_slope=0.01):
    return nn.LeakyReLU(negative_slope)


@lego("/layer/torch/prelu", alias="prelu",
      description="torch.nn.PReLU, a learned slope per channel (num_parameters) starting at init")
def prelu(num_parameters=1, init=0.25):
    return nn.PReLU(int(num_parameters), float(init))


@lego("/layer/torch/rrelu", alias="rrelu", description="torch.nn.RReLU, a random slope in [lower, upper] in train mode")
def rrelu(lower=0.125, upper=1.0 / 3.0):
    return nn.RReLU(float(lower), float(upper))


@lego("/layer/torch/elu", alias="elu", description="torch.nn.ELU with alpha")
def elu(alpha=1.0):
    return nn.ELU(float(alpha))


@lego("/layer/torch/celu", alias="celu", description="torch.nn.CELU with alpha")
def celu(alpha=1.0):
    return nn.CELU(float(alpha))


@lego("/layer/torch/selu", alias="selu", description="torch.nn.SELU, self normalizing; pairs with alpha_dropout")
def selu():
    return nn.SELU()


@lego("/layer/torch/gelu", alias="gelu", description="torch.nn.GELU; approximate none or tanh")
def gelu(approximate="none"):
    return nn.GELU(approximate=approximate)


@lego("/layer/torch/silu", alias="silu", description="torch.nn.SiLU, x times sigmoid(x)")
def silu():
    return nn.SiLU()


@lego("/layer/torch/mish", alias="mish", description="torch.nn.Mish")
def mish():
    return nn.Mish()


@lego("/layer/torch/hardswish", alias="hardswish", description="torch.nn.Hardswish")
def hardswish():
    return nn.Hardswish()


@lego("/layer/torch/hardsigmoid", alias="hardsigmoid", description="torch.nn.Hardsigmoid")
def hardsigmoid():
    return nn.Hardsigmoid()


@lego("/layer/torch/hardtanh", alias="hardtanh", description="torch.nn.Hardtanh clipped to [min_val, max_val]")
def hardtanh(min_val=-1.0, max_val=1.0):
    return nn.Hardtanh(float(min_val), float(max_val))


@lego("/layer/torch/hardshrink", alias="hardshrink", description="torch.nn.Hardshrink, zero inside [-lambd, lambd]")
def hardshrink(lambd=0.5):
    return nn.Hardshrink(float(lambd))


@lego("/layer/torch/softshrink", alias="softshrink", description="torch.nn.Softshrink, shrunk toward zero by lambd")
def softshrink(lambd=0.5):
    return nn.Softshrink(float(lambd))


@lego("/layer/torch/tanhshrink", alias="tanhshrink", description="torch.nn.Tanhshrink, x minus tanh(x)")
def tanhshrink():
    return nn.Tanhshrink()


@lego("/layer/torch/softsign", alias="softsign", description="torch.nn.Softsign, x over 1 plus |x|")
def softsign():
    return nn.Softsign()


@lego("/layer/torch/softplus", alias="softplus", description="torch.nn.Softplus with beta and the linear threshold")
def softplus(beta=1.0, threshold=20.0):
    return nn.Softplus(float(beta), float(threshold))


@lego("/layer/torch/sigmoid", alias="sigmoid", description="torch.nn.Sigmoid")
def sigmoid():
    return nn.Sigmoid()


@lego("/layer/torch/log_sigmoid", alias="log_sigmoid", description="torch.nn.LogSigmoid")
def log_sigmoid():
    return nn.LogSigmoid()


@lego("/layer/torch/tanh", alias="tanh_layer",
      description="torch.nn.Tanh; the alias is tanh_layer because tanh names the preprocessor")
def tanh():
    return nn.Tanh()


@lego("/layer/torch/threshold", alias="threshold_layer",
      description="torch.nn.Threshold: value where x is at or below threshold; the alias is threshold_layer "
                  "because threshold names the calibration")
def threshold(threshold, value):
    return nn.Threshold(float(threshold), float(value))


@lego("/layer/torch/glu", alias="glu", description="torch.nn.GLU, the gated linear unit over dim")
def glu(dim=-1):
    return nn.GLU(int(dim))


@lego("/layer/torch/softmax", alias="softmax", description="torch.nn.Softmax over dim")
def softmax(dim=-1):
    return nn.Softmax(int(dim))


@lego("/layer/torch/softmin", alias="softmin", description="torch.nn.Softmin over dim")
def softmin(dim=-1):
    return nn.Softmin(int(dim))


@lego("/layer/torch/log_softmax", alias="log_softmax", description="torch.nn.LogSoftmax over dim")
def log_softmax(dim=-1):
    return nn.LogSoftmax(int(dim))


@lego("/layer/torch/softmax2d", alias="softmax2d",
      description="torch.nn.Softmax2d, the softmax over the channels of an image")
def softmax2d():
    return nn.Softmax2d()
