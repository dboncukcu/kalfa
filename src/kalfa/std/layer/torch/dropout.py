from torch import nn

from kalfa.registration import lego


@lego("/layer/torch/dropout", alias="dropout", description="Dropout")
def dropout(p=0.5):
    return nn.Dropout(p)
