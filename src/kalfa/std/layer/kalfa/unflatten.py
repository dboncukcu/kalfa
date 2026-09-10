from torch import nn

from kalfa.registration import lego


@lego("/layer/kalfa/unflatten", alias="unflatten",
      description="Reshape the features of every sample to shape")
def unflatten(shape):
    return nn.Unflatten(1, tuple(int(part) for part in shape))
