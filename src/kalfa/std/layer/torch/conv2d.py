from torch import nn

from kalfa.registration import lego


@lego("/layer/torch/conv2d", alias="conv2d",
      description="2d convolution; without in_channels the input channels are taken from the first batch")
def conv2d(out_channels, kernel, stride=1, padding=0, in_channels=None):
    if in_channels is None:
        return nn.LazyConv2d(int(out_channels), int(kernel), stride=int(stride), padding=int(padding))
    return nn.Conv2d(int(in_channels), int(out_channels), int(kernel), stride=int(stride), padding=int(padding))
