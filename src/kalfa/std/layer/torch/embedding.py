from torch import nn

from kalfa.registration import lego
from kalfa.std.common.deferred import later


def embedding_layer(num, dim):
    return nn.Embedding(int(num), int(dim))


@lego("/layer/torch/embedding", alias="embedding",
      description="torch.nn.Embedding(num, dim); num may be a kind data component such as vocab_size")
def embedding(num, dim):
    return later(embedding_layer, num=num, dim=dim)
