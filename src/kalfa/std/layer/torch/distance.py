from torch import nn

from kalfa.registration import lego


@lego("/layer/torch/cosine_similarity", alias="cosine_similarity",
      description="torch.nn.CosineSimilarity of two inputs along dim")
def cosine_similarity(dim=1, eps=1e-8):
    return nn.CosineSimilarity(dim=int(dim), eps=float(eps))


@lego("/layer/torch/pairwise_distance", alias="pairwise_distance",
      description="torch.nn.PairwiseDistance of two inputs, the p norm of their difference")
def pairwise_distance(p=2.0, eps=1e-6, keepdim=False):
    return nn.PairwiseDistance(p=float(p), eps=float(eps), keepdim=bool(keepdim))
