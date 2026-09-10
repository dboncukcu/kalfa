from torch import nn

from kalfa.registration import lego


@lego("/layer/torch/maxpool", alias="maxpool", description="2d max pooling")
def maxpool(kernel, stride=None):
    return nn.MaxPool2d(int(kernel), stride=None if stride is None else int(stride))
