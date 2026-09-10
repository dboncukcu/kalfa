from torch import nn

from kalfa.registration import lego
from kalfa.std.common.deferred import later


def embedding_layer(num, dim, padding_idx):
    return nn.Embedding(int(num), int(dim), padding_idx=None if padding_idx is None else int(padding_idx))


def embedding_bag_layer(num, dim, mode):
    return nn.EmbeddingBag(int(num), int(dim), mode=mode)


@lego("/layer/torch/embedding", alias="embedding",
      description="torch.nn.Embedding(num, dim); num may be a kind data component such as vocab_size")
def embedding(num, dim, padding_idx=None):
    return later(embedding_layer, num=num, dim=dim, padding_idx=padding_idx)


@lego("/layer/torch/embedding_bag", alias="embedding_bag",
      description="torch.nn.EmbeddingBag(num, dim), the mean, sum or max of a bag of ids; num may be vocab_size")
def embedding_bag(num, dim, mode="mean"):
    return later(embedding_bag_layer, num=num, dim=dim, mode=mode)
