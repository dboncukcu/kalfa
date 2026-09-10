from torch import nn

from kalfa.registration import lego


def padding_of(padding):
    return int(padding) if isinstance(padding, (int, float)) else tuple(int(part) for part in padding)


@lego("/layer/torch/zero_pad", alias="zero_pad",
      description="torch.nn.ZeroPad2d; padding is a number or [left, right, top, bottom]")
def zero_pad(padding):
    return nn.ZeroPad2d(padding_of(padding))


@lego("/layer/torch/zero_pad1d", alias="zero_pad1d", description="torch.nn.ZeroPad1d")
def zero_pad1d(padding):
    return nn.ZeroPad1d(padding_of(padding))


@lego("/layer/torch/zero_pad3d", alias="zero_pad3d", description="torch.nn.ZeroPad3d")
def zero_pad3d(padding):
    return nn.ZeroPad3d(padding_of(padding))


@lego("/layer/torch/constant_pad", alias="constant_pad", description="torch.nn.ConstantPad2d with value")
def constant_pad(padding, value=0.0):
    return nn.ConstantPad2d(padding_of(padding), float(value))


@lego("/layer/torch/constant_pad1d", alias="constant_pad1d", description="torch.nn.ConstantPad1d")
def constant_pad1d(padding, value=0.0):
    return nn.ConstantPad1d(padding_of(padding), float(value))


@lego("/layer/torch/constant_pad3d", alias="constant_pad3d", description="torch.nn.ConstantPad3d")
def constant_pad3d(padding, value=0.0):
    return nn.ConstantPad3d(padding_of(padding), float(value))


@lego("/layer/torch/reflection_pad", alias="reflection_pad", description="torch.nn.ReflectionPad2d")
def reflection_pad(padding):
    return nn.ReflectionPad2d(padding_of(padding))


@lego("/layer/torch/reflection_pad1d", alias="reflection_pad1d", description="torch.nn.ReflectionPad1d")
def reflection_pad1d(padding):
    return nn.ReflectionPad1d(padding_of(padding))


@lego("/layer/torch/reflection_pad3d", alias="reflection_pad3d", description="torch.nn.ReflectionPad3d")
def reflection_pad3d(padding):
    return nn.ReflectionPad3d(padding_of(padding))


@lego("/layer/torch/replication_pad", alias="replication_pad", description="torch.nn.ReplicationPad2d")
def replication_pad(padding):
    return nn.ReplicationPad2d(padding_of(padding))


@lego("/layer/torch/replication_pad1d", alias="replication_pad1d", description="torch.nn.ReplicationPad1d")
def replication_pad1d(padding):
    return nn.ReplicationPad1d(padding_of(padding))


@lego("/layer/torch/replication_pad3d", alias="replication_pad3d", description="torch.nn.ReplicationPad3d")
def replication_pad3d(padding):
    return nn.ReplicationPad3d(padding_of(padding))


@lego("/layer/torch/circular_pad", alias="circular_pad", description="torch.nn.CircularPad2d")
def circular_pad(padding):
    return nn.CircularPad2d(padding_of(padding))


@lego("/layer/torch/circular_pad1d", alias="circular_pad1d", description="torch.nn.CircularPad1d")
def circular_pad1d(padding):
    return nn.CircularPad1d(padding_of(padding))


@lego("/layer/torch/circular_pad3d", alias="circular_pad3d", description="torch.nn.CircularPad3d")
def circular_pad3d(padding):
    return nn.CircularPad3d(padding_of(padding))
