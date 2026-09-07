"""The test stand in for a timm plugin: timm_backbone registered under the same name as a tiny convolutional net.

The width of the features follows the name (resnet50: 8, resnet18: 4, anything else: 4); pretrained is accepted and
ignored, there are no weights to load. A BatchNorm layer sits inside so freezing can be observed.
"""

import warnings

import kalfa
from torch import nn

WIDTHS = {"resnet50": 8, "resnet18": 4}


class TinyBackbone(nn.Module):
    kalfa_lazy = True

    def __init__(self, width, pooled):
        super().__init__()
        self.width = width
        self.pooled = pooled
        self.conv = nn.LazyConv2d(width, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(width)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, value):
        out = nn.functional.relu(self.norm(self.conv(value)))
        if self.pooled:
            return self.pool(out).flatten(1)
        return out


@kalfa.lego("/layer/timm/timm_backbone", alias="timm_backbone",
            description="Test backbone: one convolution, BatchNorm, ReLU and global pooling; width by name")
def timm_backbone(name, pretrained=False, pooled=True):
    if pretrained:
        warnings.warn(f"the test backbone {name!r} has no pretrained weights; training from scratch", stacklevel=2)
    return TinyBackbone(WIDTHS.get(name, 4), pooled)
