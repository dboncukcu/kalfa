import math

import torch

from kalfa.registration import lego
from kalfa.std.metric.base import Metric


class Perplexity(Metric):
    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0

    def update(self, predictions, targets):
        logits = predictions.detach().float().reshape(-1, predictions.shape[-1])
        labels = targets.reshape(-1).long()
        self.total += float(torch.nn.functional.cross_entropy(logits, labels, reduction="sum"))
        self.count += int(labels.numel())

    def compute(self):
        if not self.count:
            return math.nan
        return math.exp(self.total / self.count)


@lego("/metric/kalfa/perplexity", state=True, alias="perplexity",
      description="exp of the mean token cross entropy of the logits against the targets")
def perplexity():
    return Perplexity()
