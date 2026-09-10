from torch import nn

from kalfa.registration import lego
from kalfa.std.common.deferred import later


def layer_norm_layer(normalized_shape, eps, elementwise_affine):
    shape = int(normalized_shape) if isinstance(normalized_shape, (int, float)) else list(normalized_shape)
    return nn.LayerNorm(shape, eps=float(eps), elementwise_affine=bool(elementwise_affine))


def rms_norm_layer(normalized_shape, eps, elementwise_affine):
    shape = int(normalized_shape) if isinstance(normalized_shape, (int, float)) else list(normalized_shape)
    return nn.RMSNorm(shape, eps=None if eps is None else float(eps), elementwise_affine=bool(elementwise_affine))


@lego("/layer/torch/batch_norm", alias="batch_norm",
      description="torch.nn.BatchNorm over the feature axis, lazy in the number of features: dims 1 for (batch, "
                  "features) and sequences, 2 for images, 3 for volumes")
def batch_norm(dims=1, eps=1e-5, momentum=0.1, affine=True):
    kinds = {1: nn.LazyBatchNorm1d, 2: nn.LazyBatchNorm2d, 3: nn.LazyBatchNorm3d}
    if dims not in kinds:
        raise ValueError(f"batch_norm.dims must be 1, 2 or 3, got {dims!r}")
    return kinds[dims](eps=float(eps), momentum=float(momentum), affine=bool(affine))


@lego("/layer/torch/instance_norm", alias="instance_norm",
      description="torch.nn.InstanceNorm, lazy in the number of features: dims 1 for sequences, 2 for images, 3 "
                  "for volumes")
def instance_norm(dims=2, eps=1e-5, momentum=0.1, affine=False):
    kinds = {1: nn.LazyInstanceNorm1d, 2: nn.LazyInstanceNorm2d, 3: nn.LazyInstanceNorm3d}
    if dims not in kinds:
        raise ValueError(f"instance_norm.dims must be 1, 2 or 3, got {dims!r}")
    return kinds[dims](eps=float(eps), momentum=float(momentum), affine=bool(affine))


@lego("/layer/torch/layer_norm", alias="layer_norm",
      description="torch.nn.LayerNorm over the last axis; normalized_shape is its width, a number or "
                  "{uri: feature_width} for the width of the feature tensor")
def layer_norm(normalized_shape, eps=1e-5, elementwise_affine=True):
    return later(layer_norm_layer, normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine)


@lego("/layer/torch/rms_norm", alias="rms_norm",
      description="torch.nn.RMSNorm over the last axis; normalized_shape is its width, a number or "
                  "{uri: feature_width}")
def rms_norm(normalized_shape, eps=None, elementwise_affine=True):
    return later(rms_norm_layer, normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine)


@lego("/layer/torch/group_norm", alias="group_norm",
      description="torch.nn.GroupNorm: num_groups groups over num_channels channels")
def group_norm(num_groups, num_channels, eps=1e-5, affine=True):
    return nn.GroupNorm(int(num_groups), int(num_channels), eps=float(eps), affine=bool(affine))


@lego("/layer/torch/local_response_norm", alias="local_response_norm",
      description="torch.nn.LocalResponseNorm over size neighbouring channels")
def local_response_norm(size, alpha=1e-4, beta=0.75, k=1.0):
    return nn.LocalResponseNorm(int(size), float(alpha), float(beta), float(k))
