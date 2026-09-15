from torch import nn


def stride_of(stride):
    return None if stride is None else int(stride)


def maxpool(kernel, stride=None, padding=0):
    return nn.MaxPool2d(int(kernel), stride_of(stride), int(padding))


def maxpool1d(kernel, stride=None, padding=0):
    return nn.MaxPool1d(int(kernel), stride_of(stride), int(padding))


def maxpool3d(kernel, stride=None, padding=0):
    return nn.MaxPool3d(int(kernel), stride_of(stride), int(padding))


def avgpool(kernel, stride=None, padding=0):
    return nn.AvgPool2d(int(kernel), stride_of(stride), int(padding))


def avgpool1d(kernel, stride=None, padding=0):
    return nn.AvgPool1d(int(kernel), stride_of(stride), int(padding))


def avgpool3d(kernel, stride=None, padding=0):
    return nn.AvgPool3d(int(kernel), stride_of(stride), int(padding))


def adaptive_avgpool(output_size=1):
    return nn.AdaptiveAvgPool2d(output_size)


def adaptive_avgpool1d(output_size=1):
    return nn.AdaptiveAvgPool1d(output_size)


def adaptive_avgpool3d(output_size=1):
    return nn.AdaptiveAvgPool3d(output_size)


def adaptive_maxpool(output_size=1):
    return nn.AdaptiveMaxPool2d(output_size)


def adaptive_maxpool1d(output_size=1):
    return nn.AdaptiveMaxPool1d(output_size)


def adaptive_maxpool3d(output_size=1):
    return nn.AdaptiveMaxPool3d(output_size)


def lppool(norm_type, kernel, stride=None):
    return nn.LPPool2d(float(norm_type), int(kernel), stride_of(stride))


def lppool1d(norm_type, kernel, stride=None):
    return nn.LPPool1d(float(norm_type), int(kernel), stride_of(stride))


def lppool3d(norm_type, kernel, stride=None):
    return nn.LPPool3d(float(norm_type), int(kernel), stride_of(stride))


def fractional_maxpool(kernel, output_size=None, output_ratio=None):
    return nn.FractionalMaxPool2d(int(kernel), output_size=output_size, output_ratio=output_ratio)


def fractional_maxpool3d(kernel, output_size=None, output_ratio=None):
    return nn.FractionalMaxPool3d(int(kernel), output_size=output_size, output_ratio=output_ratio)


def max_unpool(kernel, stride=None, padding=0):
    return nn.MaxUnpool2d(int(kernel), stride_of(stride), int(padding))


def max_unpool1d(kernel, stride=None, padding=0):
    return nn.MaxUnpool1d(int(kernel), stride_of(stride), int(padding))


def max_unpool3d(kernel, stride=None, padding=0):
    return nn.MaxUnpool3d(int(kernel), stride_of(stride), int(padding))
