from torch import nn

from kalfa.registration import lego


@lego("/layer/torch/dropout", alias="dropout", description="torch.nn.Dropout")
def dropout(p=0.5):
    return nn.Dropout(p)


@lego("/layer/torch/dropout1d", alias="dropout1d", description="torch.nn.Dropout1d, whole channels of a sequence")
def dropout1d(p=0.5):
    return nn.Dropout1d(p)


@lego("/layer/torch/dropout2d", alias="dropout2d", description="torch.nn.Dropout2d, whole channels of an image")
def dropout2d(p=0.5):
    return nn.Dropout2d(p)


@lego("/layer/torch/dropout3d", alias="dropout3d", description="torch.nn.Dropout3d, whole channels of a volume")
def dropout3d(p=0.5):
    return nn.Dropout3d(p)


@lego("/layer/torch/alpha_dropout", alias="alpha_dropout", description="torch.nn.AlphaDropout, the dropout of selu")
def alpha_dropout(p=0.5):
    return nn.AlphaDropout(p)


@lego("/layer/torch/feature_alpha_dropout", alias="feature_alpha_dropout",
      description="torch.nn.FeatureAlphaDropout, alpha dropout of whole channels")
def feature_alpha_dropout(p=0.5):
    return nn.FeatureAlphaDropout(p)
