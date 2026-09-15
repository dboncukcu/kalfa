from torch import nn


def dropout(p=0.5):
    return nn.Dropout(p)


def dropout1d(p=0.5):
    return nn.Dropout1d(p)


def dropout2d(p=0.5):
    return nn.Dropout2d(p)


def dropout3d(p=0.5):
    return nn.Dropout3d(p)


def alpha_dropout(p=0.5):
    return nn.AlphaDropout(p)


def feature_alpha_dropout(p=0.5):
    return nn.FeatureAlphaDropout(p)
