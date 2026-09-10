from kalfa.registration import lego
from kalfa.std.common.deferred import later
from kalfa.std.layer.base import linear_layer


@lego("/layer/kalfa/linear", alias="linear",
      description="Linear layer; without in_features the input width is taken from the first batch")
def linear(out_features, in_features=None):
    return later(linear_layer, out_features=out_features, in_features=in_features)
