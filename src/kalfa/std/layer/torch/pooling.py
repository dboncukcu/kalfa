from torch import nn

from kalfa.registration import lego


def stride_of(stride):
    return None if stride is None else int(stride)


@lego("/layer/torch/maxpool", alias="maxpool", description="torch.nn.MaxPool2d")
def maxpool(kernel, stride=None, padding=0):
    return nn.MaxPool2d(int(kernel), stride_of(stride), int(padding))


@lego("/layer/torch/maxpool1d", alias="maxpool1d", description="torch.nn.MaxPool1d")
def maxpool1d(kernel, stride=None, padding=0):
    return nn.MaxPool1d(int(kernel), stride_of(stride), int(padding))


@lego("/layer/torch/maxpool3d", alias="maxpool3d", description="torch.nn.MaxPool3d")
def maxpool3d(kernel, stride=None, padding=0):
    return nn.MaxPool3d(int(kernel), stride_of(stride), int(padding))


@lego("/layer/torch/avgpool", alias="avgpool", description="torch.nn.AvgPool2d")
def avgpool(kernel, stride=None, padding=0):
    return nn.AvgPool2d(int(kernel), stride_of(stride), int(padding))


@lego("/layer/torch/avgpool1d", alias="avgpool1d", description="torch.nn.AvgPool1d")
def avgpool1d(kernel, stride=None, padding=0):
    return nn.AvgPool1d(int(kernel), stride_of(stride), int(padding))


@lego("/layer/torch/avgpool3d", alias="avgpool3d", description="torch.nn.AvgPool3d")
def avgpool3d(kernel, stride=None, padding=0):
    return nn.AvgPool3d(int(kernel), stride_of(stride), int(padding))


@lego("/layer/torch/adaptive_avgpool", alias="adaptive_avgpool",
      description="torch.nn.AdaptiveAvgPool2d to output_size, a number or [height, width]")
def adaptive_avgpool(output_size=1):
    return nn.AdaptiveAvgPool2d(output_size)


@lego("/layer/torch/adaptive_avgpool1d", alias="adaptive_avgpool1d", description="torch.nn.AdaptiveAvgPool1d")
def adaptive_avgpool1d(output_size=1):
    return nn.AdaptiveAvgPool1d(output_size)


@lego("/layer/torch/adaptive_avgpool3d", alias="adaptive_avgpool3d", description="torch.nn.AdaptiveAvgPool3d")
def adaptive_avgpool3d(output_size=1):
    return nn.AdaptiveAvgPool3d(output_size)


@lego("/layer/torch/adaptive_maxpool", alias="adaptive_maxpool",
      description="torch.nn.AdaptiveMaxPool2d to output_size, a number or [height, width]")
def adaptive_maxpool(output_size=1):
    return nn.AdaptiveMaxPool2d(output_size)


@lego("/layer/torch/adaptive_maxpool1d", alias="adaptive_maxpool1d", description="torch.nn.AdaptiveMaxPool1d")
def adaptive_maxpool1d(output_size=1):
    return nn.AdaptiveMaxPool1d(output_size)


@lego("/layer/torch/adaptive_maxpool3d", alias="adaptive_maxpool3d", description="torch.nn.AdaptiveMaxPool3d")
def adaptive_maxpool3d(output_size=1):
    return nn.AdaptiveMaxPool3d(output_size)


@lego("/layer/torch/lppool", alias="lppool", description="torch.nn.LPPool2d, the power average pool of norm_type")
def lppool(norm_type, kernel, stride=None):
    return nn.LPPool2d(float(norm_type), int(kernel), stride_of(stride))


@lego("/layer/torch/lppool1d", alias="lppool1d", description="torch.nn.LPPool1d")
def lppool1d(norm_type, kernel, stride=None):
    return nn.LPPool1d(float(norm_type), int(kernel), stride_of(stride))


@lego("/layer/torch/lppool3d", alias="lppool3d", description="torch.nn.LPPool3d")
def lppool3d(norm_type, kernel, stride=None):
    return nn.LPPool3d(float(norm_type), int(kernel), stride_of(stride))


@lego("/layer/torch/fractional_maxpool", alias="fractional_maxpool",
      description="torch.nn.FractionalMaxPool2d to output_size or output_ratio")
def fractional_maxpool(kernel, output_size=None, output_ratio=None):
    return nn.FractionalMaxPool2d(int(kernel), output_size=output_size, output_ratio=output_ratio)


@lego("/layer/torch/fractional_maxpool3d", alias="fractional_maxpool3d", description="torch.nn.FractionalMaxPool3d")
def fractional_maxpool3d(kernel, output_size=None, output_ratio=None):
    return nn.FractionalMaxPool3d(int(kernel), output_size=output_size, output_ratio=output_ratio)


@lego("/layer/torch/max_unpool", alias="max_unpool",
      description="torch.nn.MaxUnpool2d, the inverse of a max pool that kept its indices; two inputs")
def max_unpool(kernel, stride=None, padding=0):
    return nn.MaxUnpool2d(int(kernel), stride_of(stride), int(padding))


@lego("/layer/torch/max_unpool1d", alias="max_unpool1d", description="torch.nn.MaxUnpool1d; two inputs")
def max_unpool1d(kernel, stride=None, padding=0):
    return nn.MaxUnpool1d(int(kernel), stride_of(stride), int(padding))


@lego("/layer/torch/max_unpool3d", alias="max_unpool3d", description="torch.nn.MaxUnpool3d; two inputs")
def max_unpool3d(kernel, stride=None, padding=0):
    return nn.MaxUnpool3d(int(kernel), stride_of(stride), int(padding))
