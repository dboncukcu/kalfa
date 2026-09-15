from torch import nn


def padding_of(padding):
    return int(padding) if isinstance(padding, (int, float)) else tuple(int(part) for part in padding)


def zero_pad(padding):
    return nn.ZeroPad2d(padding_of(padding))


def zero_pad1d(padding):
    return nn.ZeroPad1d(padding_of(padding))


def zero_pad3d(padding):
    return nn.ZeroPad3d(padding_of(padding))


def constant_pad(padding, value=0.0):
    return nn.ConstantPad2d(padding_of(padding), float(value))


def constant_pad1d(padding, value=0.0):
    return nn.ConstantPad1d(padding_of(padding), float(value))


def constant_pad3d(padding, value=0.0):
    return nn.ConstantPad3d(padding_of(padding), float(value))


def reflection_pad(padding):
    return nn.ReflectionPad2d(padding_of(padding))


def reflection_pad1d(padding):
    return nn.ReflectionPad1d(padding_of(padding))


def reflection_pad3d(padding):
    return nn.ReflectionPad3d(padding_of(padding))


def replication_pad(padding):
    return nn.ReplicationPad2d(padding_of(padding))


def replication_pad1d(padding):
    return nn.ReplicationPad1d(padding_of(padding))


def replication_pad3d(padding):
    return nn.ReplicationPad3d(padding_of(padding))


def circular_pad(padding):
    return nn.CircularPad2d(padding_of(padding))


def circular_pad1d(padding):
    return nn.CircularPad1d(padding_of(padding))


def circular_pad3d(padding):
    return nn.CircularPad3d(padding_of(padding))
