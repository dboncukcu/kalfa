import torch
from torch import nn

from kalfa.registration import lego


@lego("/layer/torch/concat", alias="concat", description="Concatenate wires along a dimension")
class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, *values):
        return torch.cat(values, dim=self.dim)


@lego("/layer/torch/last_step", alias="last_step", description="The last step of a sequence")
class LastStep(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, value):
        return value[:, -1, :]


@lego("/layer/torch/flatten", alias="flatten", description="torch.nn.Flatten from start_dim to end_dim")
def flatten(start_dim=1, end_dim=-1):
    return nn.Flatten(int(start_dim), int(end_dim))


@lego("/layer/torch/unflatten",
      description="torch.nn.Unflatten of dim into size; the alias unflatten names kalfa's per sample reshape")
def unflatten(dim, size):
    return nn.Unflatten(int(dim), tuple(int(part) for part in size))


@lego("/layer/torch/fold", alias="fold", description="torch.nn.Fold, sliding blocks back into an image of output_size")
def fold(output_size, kernel, dilation=1, padding=0, stride=1):
    return nn.Fold(output_size, kernel, dilation=dilation, padding=padding, stride=stride)


@lego("/layer/torch/unfold", alias="unfold", description="torch.nn.Unfold, an image into sliding blocks")
def unfold(kernel, dilation=1, padding=0, stride=1):
    return nn.Unfold(kernel, dilation=dilation, padding=padding, stride=stride)


@lego("/layer/torch/pixel_shuffle", alias="pixel_shuffle", description="torch.nn.PixelShuffle by factor")
def pixel_shuffle(factor):
    return nn.PixelShuffle(int(factor))


@lego("/layer/torch/pixel_unshuffle", alias="pixel_unshuffle", description="torch.nn.PixelUnshuffle by factor")
def pixel_unshuffle(factor):
    return nn.PixelUnshuffle(int(factor))


@lego("/layer/torch/channel_shuffle", alias="channel_shuffle", description="torch.nn.ChannelShuffle over groups")
def channel_shuffle(groups):
    return nn.ChannelShuffle(int(groups))


@lego("/layer/torch/upsample", alias="upsample",
      description="torch.nn.Upsample to size or by scale_factor; mode nearest, linear, bilinear, bicubic or "
                  "trilinear")
def upsample(size=None, scale_factor=None, mode="nearest", align_corners=None):
    return nn.Upsample(size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)
