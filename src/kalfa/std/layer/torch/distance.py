from torch import nn


def cosine_similarity(dim=1, eps=1e-8):
    return nn.CosineSimilarity(dim=int(dim), eps=float(eps))


def pairwise_distance(p=2.0, eps=1e-6, keepdim=False):
    return nn.PairwiseDistance(p=float(p), eps=float(eps), keepdim=bool(keepdim))
