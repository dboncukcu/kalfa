from torch import nn

from kalfa.registration import lego
from kalfa.std.common.deferred import later
from kalfa.std.layer.base import linear_layer


@lego("/layer/kalfa/linear_relu", alias="linear_relu",
      description="Linear layer followed by ReLU; lazy without in_features")
def linear_relu(out_features, in_features=None):
    return nn.Sequential(later(linear_layer, out_features=out_features, in_features=in_features), nn.ReLU())
