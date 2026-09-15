import math

from kalfa.std.metric.base import Metric


class ReconError(Metric):
    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0

    def update(self, predictions, targets):
        predictions = predictions.reshape(len(predictions), -1).float()
        targets = targets.reshape(len(targets), -1).float()
        self.total += float(((predictions - targets) ** 2).mean(dim=1).sum())
        self.count += len(targets)

    def compute(self):
        if not self.count:
            return math.nan
        return self.total / self.count
