import math

from kalfa.std.metric.base import Metric


class Rmse(Metric):
    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0

    def update(self, predictions, targets):
        predictions = predictions.reshape(len(predictions), -1).float()
        targets = targets.reshape(len(targets), -1).float()
        self.total += float(((predictions - targets) ** 2).sum())
        self.count += targets.numel()

    def compute(self):
        if not self.count:
            return math.nan
        return math.sqrt(self.total / self.count)
