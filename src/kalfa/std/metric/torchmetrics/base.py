import math
import warnings

import torch

from kalfa.std.metric.base import Metric


class TorchMetric(Metric):
    def __init__(self, factory, name):
        self.factory = factory
        self.name = name
        self.metric = factory()
        self.warned = False
        self.classes = set()

    def reset(self):
        self.metric.reset()
        self.classes = set()

    def update(self, predictions, targets):
        labels = targets.reshape(-1).long()
        self.classes.update(int(value) for value in torch.unique(labels).tolist())
        self.metric.update(predictions.detach().reshape(-1).float(), labels)

    def compute(self):
        value = math.nan
        if len(self.classes) >= 2:
            try:
                value = float(self.metric.compute())
            except (ValueError, RuntimeError, IndexError):
                value = math.nan
        if math.isnan(value) and not self.warned:
            warnings.warn(f"{self.name} is undefined on a set with fewer than two classes; reported as NaN",
                          stacklevel=2)
            self.warned = True
        return value

    def __deepcopy__(self, memo):
        copy = TorchMetric(self.factory, self.name)
        copy.warned = self.warned
        return copy


class ClassMetric(Metric):
    def __init__(self, name, average):
        self.name = name
        self.average = average
        self.reset()

    def reset(self):
        self.predictions = []
        self.targets = []

    def update(self, predictions, targets):
        scores = predictions.detach().cpu()
        if scores.ndim == 1 or scores.shape[-1] == 1:
            picked = (scores.reshape(-1) > 0).long()
        else:
            picked = scores.reshape(len(scores), -1).argmax(dim=-1)
        self.predictions.append(picked)
        self.targets.append(targets.detach().cpu().reshape(-1).long())

    def compute(self):
        from torchmetrics.functional.classification import multiclass_accuracy, multiclass_f1_score

        if not self.predictions:
            return math.nan
        predictions = torch.cat(self.predictions)
        targets = torch.cat(self.targets)
        classes = int(max(int(predictions.max()), int(targets.max())) + 1)
        if classes < 2:
            classes = 2
        if self.name == "accuracy":
            return float(multiclass_accuracy(predictions, targets, num_classes=classes, average="micro"))
        return float(multiclass_f1_score(predictions, targets, num_classes=classes, average=self.average))
