from torch import nn

from kalfa.registration import lego
from kalfa.std.common.deferred import later


def layer_norm_layer(normalized_shape, eps, elementwise_affine):
    shape = int(normalized_shape) if isinstance(normalized_shape, (int, float)) else list(normalized_shape)
    return nn.LayerNorm(shape, eps=float(eps), elementwise_affine=bool(elementwise_affine))


@lego("/layer/torch/layer_norm", alias="layer_norm",
      description="torch.nn.LayerNorm over the last axis; normalized_shape is its width, a number or "
                  "{uri: feature_width} for the width of the feature tensor")
def layer_norm(normalized_shape, eps=1e-5, elementwise_affine=True):
    return later(layer_norm_layer, normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
