from torch import nn


def zeros():
    def apply(tensor):
        nn.init.zeros_(tensor)
    return apply


def normal(std, mean=0.0):
    def apply(tensor):
        nn.init.normal_(tensor, mean=mean, std=std)
    return apply


def xavier(gain=1.0):
    def apply(tensor):
        nn.init.xavier_uniform_(tensor, gain=gain)
    return apply


def kaiming(nonlinearity="relu"):
    def apply(tensor):
        nn.init.kaiming_normal_(tensor, nonlinearity=nonlinearity)
    return apply


def kaiming_uniform(nonlinearity="relu"):
    def apply(tensor):
        nn.init.kaiming_uniform_(tensor, nonlinearity=nonlinearity)
    return apply
