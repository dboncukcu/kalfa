from torch import nn

from kalfa.registration import lego


def size_of(value):
    return int(value) if isinstance(value, (int, float)) else tuple(int(part) for part in value)


def convolution(kinds, in_channels, out_channels, kernel, stride, padding, dilation, groups, bias):
    if in_channels is None:
        return kinds[0](int(out_channels), size_of(kernel), stride=size_of(stride), padding=padding,
                        dilation=size_of(dilation), groups=int(groups), bias=bool(bias))
    return kinds[1](int(in_channels), int(out_channels), size_of(kernel), stride=size_of(stride), padding=padding,
                    dilation=size_of(dilation), groups=int(groups), bias=bool(bias))


def transposed(kinds, in_channels, out_channels, kernel, stride, padding, output_padding, dilation, groups, bias):
    if in_channels is None:
        return kinds[0](int(out_channels), size_of(kernel), stride=size_of(stride), padding=size_of(padding),
                        output_padding=size_of(output_padding), dilation=size_of(dilation), groups=int(groups),
                        bias=bool(bias))
    return kinds[1](int(in_channels), int(out_channels), size_of(kernel), stride=size_of(stride),
                    padding=size_of(padding), output_padding=size_of(output_padding), dilation=size_of(dilation),
                    groups=int(groups), bias=bool(bias))


@lego("/layer/torch/conv1d", alias="conv1d",
      description="torch.nn.Conv1d; without in_channels the input channels are taken from the first batch")
def conv1d(out_channels, kernel, stride=1, padding=0, dilation=1, groups=1, bias=True, in_channels=None):
    return convolution((nn.LazyConv1d, nn.Conv1d), in_channels, out_channels, kernel, stride, padding, dilation,
                       groups, bias)


@lego("/layer/torch/conv2d", alias="conv2d",
      description="2d convolution; without in_channels the input channels are taken from the first batch")
def conv2d(out_channels, kernel, stride=1, padding=0, dilation=1, groups=1, bias=True, in_channels=None):
    return convolution((nn.LazyConv2d, nn.Conv2d), in_channels, out_channels, kernel, stride, padding, dilation,
                       groups, bias)


@lego("/layer/torch/conv3d", alias="conv3d",
      description="torch.nn.Conv3d; without in_channels the input channels are taken from the first batch")
def conv3d(out_channels, kernel, stride=1, padding=0, dilation=1, groups=1, bias=True, in_channels=None):
    return convolution((nn.LazyConv3d, nn.Conv3d), in_channels, out_channels, kernel, stride, padding, dilation,
                       groups, bias)


@lego("/layer/torch/conv_transpose1d", alias="conv_transpose1d",
      description="torch.nn.ConvTranspose1d; without in_channels the input channels are taken from the first batch")
def conv_transpose1d(out_channels, kernel, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True,
                     in_channels=None):
    return transposed((nn.LazyConvTranspose1d, nn.ConvTranspose1d), in_channels, out_channels, kernel, stride,
                      padding, output_padding, dilation, groups, bias)


@lego("/layer/torch/conv_transpose2d", alias="conv_transpose2d",
      description="torch.nn.ConvTranspose2d; without in_channels the input channels are taken from the first batch")
def conv_transpose2d(out_channels, kernel, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True,
                     in_channels=None):
    return transposed((nn.LazyConvTranspose2d, nn.ConvTranspose2d), in_channels, out_channels, kernel, stride,
                      padding, output_padding, dilation, groups, bias)


@lego("/layer/torch/conv_transpose3d", alias="conv_transpose3d",
      description="torch.nn.ConvTranspose3d; without in_channels the input channels are taken from the first batch")
def conv_transpose3d(out_channels, kernel, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True,
                     in_channels=None):
    return transposed((nn.LazyConvTranspose3d, nn.ConvTranspose3d), in_channels, out_channels, kernel, stride,
                      padding, output_padding, dilation, groups, bias)
