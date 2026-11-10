import math

import numpy
import torch
from torch.nn import functional

from kalfa.core.aliases import import_target, register
from kalfa.core.errors import BuildError, RunError
from kalfa.core.protocols import ContextMetric
from kalfa.std.common import classes_last_of, evaluating


def flat(value):
    return value.detach().cpu().reshape(-1).to(torch.float64)


def pair(pred, target):
    guess, truth = flat(pred), flat(target)
    if guess.numel() != truth.numel():
        raise ValueError(f"pred has {guess.numel()} values and target {truth.numel()}; the metric needs one per row")
    return guess, truth


def labels_of(pred):
    scores = pred.detach().cpu()
    if scores.ndim >= 2 and scores.shape[-1] > 1:
        return scores.reshape(len(scores), -1).argmax(dim=-1)
    scores = scores.reshape(-1)
    if scores.is_floating_point():
        return (scores > 0).long()
    return scores.long()


def scores_of(pred):
    scores = pred.detach().cpu().to(torch.float32)
    if scores.ndim <= 1 or (scores.ndim == 2 and scores.shape[1] == 1):
        return scores.reshape(-1)
    scores = torch.softmax(scores.reshape(len(scores), -1), dim=1)
    return scores[:, 1] if scores.shape[1] == 2 else scores


def targets_of(target, pred):
    truth = target.detach().cpu()
    if truth.ndim >= 2 and truth.shape[-1] > 1 and tuple(truth.shape) == tuple(pred.shape):
        return truth.reshape(len(truth), -1).argmax(dim=-1)
    return truth.reshape(-1).long()


def check_rows(rows, truth):
    if rows != truth.numel():
        raise ValueError(f"pred has {rows} rows and target {truth.numel()}")


class SumMetric:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def reset(self):
        self.total = 0.0
        self.count = 0

    def to(self, device):
        return self

    def finish(self, mean):
        return mean

    def compute(self):
        if not self.count:
            return math.nan
        return float(self.finish(self.total / self.count))


class Rmse(SumMetric):
    def update(self, pred, target):
        guess, truth = pair(pred, target)
        self.total += float(((guess - truth) ** 2).sum())
        self.count += int(truth.numel())

    def finish(self, mean):
        return math.sqrt(mean)


class Mae(SumMetric):
    def update(self, pred, target):
        guess, truth = pair(pred, target)
        self.total += float((guess - truth).abs().sum())
        self.count += int(truth.numel())


class Confusion:
    averages = ()

    def __init__(self, num_classes=None, average=None):
        if average not in self.averages:
            raise ValueError(f"average must be one of {list(self.averages)}, got {average!r}")
        self.num_classes = None if num_classes is None else int(num_classes)
        self.average = average
        self.counts = {}

    def reset(self):
        self.counts = {}

    def to(self, device):
        return self

    def update(self, pred, target):
        picked = labels_of(pred)
        truth = targets_of(target, pred)
        check_rows(picked.numel(), truth)
        if picked.numel() == 0:
            return
        pairs, counts = torch.unique(torch.stack([truth, picked], dim=1), dim=0, return_counts=True)
        for (seen, guessed), count in zip(pairs.tolist(), counts.tolist()):
            self.counts[(seen, guessed)] = self.counts.get((seen, guessed), 0) + count

    def classes(self):
        if self.num_classes is not None:
            return list(range(self.num_classes))
        return sorted({label for entry in self.counts for label in entry})

    def total(self):
        return sum(self.counts.values())

    def support(self, label):
        return sum(count for (seen, _), count in self.counts.items() if seen == label)

    def predicted(self, label):
        return sum(count for (_, guessed), count in self.counts.items() if guessed == label)

    def hits(self, label):
        return self.counts.get((label, label), 0)


class Accuracy(Confusion):
    averages = ("micro", "macro")

    def __init__(self, num_classes=None, average="micro"):
        super().__init__(num_classes, average)

    def compute(self):
        total = self.total()
        if not total:
            return math.nan
        if self.average == "micro":
            return sum(self.hits(label) for label in self.classes()) / total
        recalls = [self.hits(label) / self.support(label) for label in self.classes() if self.support(label)]
        return sum(recalls) / len(recalls)


class F1(Confusion):
    averages = ("macro", "micro", "weighted", "binary")

    def __init__(self, num_classes=None, average="macro"):
        super().__init__(num_classes, average)

    def score(self, label):
        denominator = self.predicted(label) + self.support(label)
        return 2.0 * self.hits(label) / denominator if denominator else 0.0

    def compute(self):
        total = self.total()
        if not total:
            return math.nan
        if self.average == "binary":
            return self.score(1)
        if self.average == "micro":
            return sum(self.hits(label) for label in self.classes()) / total
        labels = self.classes()
        if self.average == "weighted":
            return sum(self.score(label) * self.support(label) for label in labels) / total
        return sum(self.score(label) for label in labels) / len(labels)


class Scored:
    def __init__(self, num_classes=None):
        self.num_classes = None if num_classes is None else int(num_classes)
        self.scores = []
        self.targets = []

    def reset(self):
        self.scores = []
        self.targets = []

    def to(self, device):
        return self

    def update(self, pred, target):
        scores = scores_of(pred)
        truth = targets_of(target, pred)
        check_rows(len(scores), truth)
        if truth.numel() == 0:
            return
        self.scores.append(scores)
        self.targets.append(truth)

    def gathered(self):
        if not self.scores:
            return None, None
        return torch.cat(self.scores).numpy(), torch.cat(self.targets).numpy()

    @staticmethod
    def scorable(truth):
        return len(truth) > 0 and truth.min() != truth.max()

    def compute(self):
        scores, truth = self.gathered()
        if scores is None:
            return math.nan
        if scores.ndim == 1:
            return float(self.binary(scores, truth)) if self.scorable(truth) else math.nan
        values = []
        for label in range(scores.shape[1]):
            positives = (truth == label).astype(int)
            if self.scorable(positives):
                values.append(float(self.binary(scores[:, label], positives)))
        return sum(values) / len(values) if values else math.nan


class Auroc(Scored):
    def binary(self, scores, truth):
        from sklearn.metrics import roc_auc_score

        return roc_auc_score(truth, scores)


class AveragePrecision(Scored):
    def binary(self, scores, truth):
        from sklearn.metrics import average_precision_score

        return average_precision_score(truth, scores)


def rmse():
    return Rmse()


def mae():
    return Mae()


def accuracy(num_classes=None, average="micro"):
    return Accuracy(num_classes, average)


def f1(num_classes=None, average="macro"):
    return F1(num_classes, average)


def auroc(num_classes=None):
    return Auroc(num_classes)


def average_precision(num_classes=None):
    return AveragePrecision(num_classes)


METRICS = (
    ("rmse", "kalfa.std.metrics.rmse", "root mean squared error over all predicted values"),
    ("mae", "kalfa.std.metrics.mae", "mean absolute error over all predicted values"),
    ("accuracy", "kalfa.std.metrics.accuracy", "fraction of rows whose predicted class (argmax of logits, or "
                                               "logit > 0) matches the target; average micro or macro"),
    ("f1", "kalfa.std.metrics.f1", "f1 over the classes seen (or num_classes); average macro, micro, weighted or "
                                   "binary"),
    ("auroc", "kalfa.std.metrics.auroc", "area under the roc curve from scores, (N, 2) logits or one versus rest "
                                         "macro over (N, C) logits; nan when the set holds one class"),
    ("average_precision", "kalfa.std.metrics.average_precision", "average precision from scores, (N, 2) logits "
                                                                 "or one versus rest macro over (N, C) logits"),
)


def register_names():
    for name, target, doc in METRICS:
        register(name, target, slot="metric", doc=doc)


register_names()


class Perplexity:
    def __init__(self, classes_last=None):
        self.classes_last = classes_last
        self.reset()

    def reset(self):
        self.total = 0.0
        self.tokens = 0

    def to(self, device):
        return self

    def update(self, pred, target):
        logits = pred.detach().cpu().to(torch.float32)
        truth = target.detach().cpu().long()
        if logits.ndim < 2:
            raise ValueError(f"perplexity needs logits (N, T, V) or (N, V), got shape {tuple(logits.shape)}")
        if classes_last_of(logits, truth, self.classes_last, self, "perplexity") is False and logits.ndim >= 3:
            logits = logits.movedim(1, -1)
        logits = logits.reshape(-1, logits.shape[-1])
        truth = truth.reshape(-1)
        if truth.numel() != len(logits):
            raise ValueError(f"perplexity has logits of shape {tuple(pred.shape)} for targets of shape "
                             f"{tuple(target.shape)}; it needs (N, T, V) or (N, V, T) logits for (N, T) targets")
        counted = int((truth != -100).sum())
        if not counted:
            return
        self.total += float(functional.cross_entropy(logits, truth, reduction="sum", ignore_index=-100))
        self.tokens += counted

    def compute(self):
        if not self.tokens:
            return math.nan
        try:
            return math.exp(self.total / self.tokens)
        except OverflowError:
            return math.inf


def pixels(images):
    values = images.detach().cpu().to(torch.float64)
    if values.ndim == 3:
        values = values.unsqueeze(1)
    if values.ndim != 4:
        raise RunError(f"the pixel extractor needs images (N, C, H, W), got shape {tuple(values.shape)}")
    return functional.adaptive_avg_pool2d(values, 16).flatten(1)


def extractor_of(spec):
    if spec is None:
        return pixels
    found = spec
    if isinstance(spec, str):
        try:
            found = import_target(spec)
        except BuildError as failure:
            raise BuildError(f"fid: extractor {spec!r} is not an importable function images -> features "
                             f"({failure})") from None
    if not callable(found):
        raise BuildError(f"fid: extractor {spec!r} is not a function images -> features")
    return found


def feature_rows(values):
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().to(torch.float64).numpy()
    values = numpy.asarray(values, dtype=numpy.float64)
    return values.reshape(len(values), -1)


def first_tensor(value):
    if isinstance(value, dict):
        value = next(iter(value.values()))
    if isinstance(value, (tuple, list)):
        value = value[0]
    return value


def symmetric(matrix):
    return (matrix + matrix.T) / 2.0


def moments(rows):
    width = rows.shape[1]
    return rows.mean(axis=0), numpy.cov(rows, rowvar=False).reshape(width, width)


def root_of(matrix):
    values, vectors = numpy.linalg.eigh(symmetric(matrix))
    return (vectors * numpy.sqrt(numpy.clip(values, 0.0, None))) @ vectors.T


def frechet_distance(real, fake):
    real = numpy.asarray(real, dtype=numpy.float64)
    fake = numpy.asarray(fake, dtype=numpy.float64)
    if len(real) < 2 or len(fake) < 2:
        return math.nan
    real = real.reshape(len(real), -1)
    fake = fake.reshape(len(fake), -1)
    if real.shape[1] != fake.shape[1]:
        raise RunError(f"fid: the real features have {real.shape[1]} columns and the fakes {fake.shape[1]}")
    if not (numpy.isfinite(real).all() and numpy.isfinite(fake).all()):
        return math.nan
    mean_real, cov_real = moments(real)
    mean_fake, cov_fake = moments(fake)
    root = root_of(cov_real)
    values = numpy.linalg.eigvalsh(symmetric(root @ cov_fake @ root))
    shared = numpy.sqrt(numpy.clip(values, 0.0, None)).sum()
    difference = mean_real - mean_fake
    distance = difference @ difference + numpy.trace(cov_real) + numpy.trace(cov_fake) - 2.0 * shared
    return max(float(distance), 0.0)


class Fid(ContextMetric):
    on = ("valid",)

    def __init__(self, model, latent, real="image", label=None, n=2000, extractor=None, batch=128):
        self.model = model
        self.latent = int(latent)
        self.real = real
        self.label = label
        self.n = int(n)
        self.extractor = extractor
        self.batch = int(batch)
        if self.batch < 1:
            raise BuildError(f"fid: batch must be a positive int, got {batch!r}")
        self.extract = extractor_of(extractor)
        self.reset()

    def reset(self):
        self.real_features = []
        self.fake_features = []
        self.seen = 0

    def field(self, batch, name):
        if name not in batch:
            raise RunError(f"{self.path or 'fid'}: the batch has no field {name!r}; it has {sorted(batch)}")
        return batch[name]

    def features_of(self, images):
        return feature_rows(self.extract(images))

    def fakes(self, noise, labels):
        found = []
        for start in range(0, len(noise), self.batch):
            parts = [noise[start:start + self.batch]]
            if labels is not None:
                parts.append(labels[start:start + self.batch])
            found.append(self.features_of(first_tensor(self.model(*parts))))
        return found

    def update(self, outputs, batch, ctx):
        need = self.n - self.seen
        if need <= 0:
            return
        real = self.field(batch, self.real)[:need]
        count = len(real)
        if not count:
            return
        labels = None if self.label is None else self.field(batch, self.label)[:count]
        with evaluating(self.model):
            self.real_features.append(self.features_of(real))
            noise = ctx.rng.normal("fid", (count, self.latent), like=real)
            self.fake_features.extend(self.fakes(noise, labels))
        self.seen += count

    def compute(self):
        if not self.real_features or not self.fake_features:
            return math.nan
        return frechet_distance(numpy.concatenate(self.real_features), numpy.concatenate(self.fake_features))


def perplexity(classes_last=None):
    return Perplexity(classes_last)


register("perplexity", "kalfa.std.metrics.perplexity", slot="metric",
         doc="exp of the mean token cross entropy of logits (N, T, V), (N, V, T) or (N, V) against int targets; "
             "classes_last: true or false states the layout when the tokens and the vocabulary have the same size")
register("fid", "kalfa.std.metrics.Fid", slot="metric",
         doc="Frechet distance between features of n real images and n images of model(z) (or model(z, label)); "
             "extractor is an import path of a function images -> features, the default pools pixels; valid only, "
             "pair it with every to keep it cheap")
