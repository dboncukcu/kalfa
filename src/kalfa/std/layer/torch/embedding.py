from torch import nn

from kalfa.std.common.deferred import later


def embedding_layer(num, dim, padding_idx):
    return nn.Embedding(int(num), int(dim), padding_idx=None if padding_idx is None else int(padding_idx))


def embedding_bag_layer(num, dim, mode):
    return nn.EmbeddingBag(int(num), int(dim), mode=mode)


def embedding(num, dim, padding_idx=None):
    return later(embedding_layer, num=num, dim=dim, padding_idx=padding_idx)


def embedding_bag(num, dim, mode="mean"):
    return later(embedding_bag_layer, num=num, dim=dim, mode=mode)
