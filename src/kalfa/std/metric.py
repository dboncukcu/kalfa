"""Metrics: stateful update/compute legos."""

import math

import torch

from ..registration import lego


class Rmse:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0

    def update(self, predictions, targets):
        predictions = predictions.reshape(len(predictions), -1).double()
        targets = targets.reshape(len(targets), -1).double()
        self.total += float(((predictions - targets) ** 2).sum())
        self.count += targets.numel()

    def compute(self):
        if not self.count:
            return math.nan
        return math.sqrt(self.total / self.count)


@lego("/metric/kalfa/rmse", state=True, alias="rmse", description="Root mean squared error")
def rmse():
    return Rmse()


class ReconError:
    """The mean over samples of the per sample mean squared error between the output and the target."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0

    def update(self, predictions, targets):
        predictions = predictions.reshape(len(predictions), -1).double()
        targets = targets.reshape(len(targets), -1).double()
        self.total += float(((predictions - targets) ** 2).mean(dim=1).sum())
        self.count += len(targets)

    def compute(self):
        if not self.count:
            return math.nan
        return self.total / self.count


@lego("/metric/kalfa/recon_error", state=True, alias="recon_error",
            description="Mean per sample squared reconstruction error of the output against the target")
def recon_error():
    return ReconError()


def as_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach())
    return float(value)


class TorchMetric:
    """A torchmetrics metric under kalfa's interface; an undefined value is NaN with one warning."""

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
        import warnings

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


@lego("/metric/torchmetrics/binary_auroc", state=True, alias="auroc",
            description="Area under the ROC curve of binary scores (torchmetrics)")
def binary_auroc():
    from torchmetrics.classification import BinaryAUROC

    return TorchMetric(BinaryAUROC, "auroc")


@lego("/metric/torchmetrics/binary_average_precision", state=True, alias="average_precision",
            description="Average precision of binary scores (torchmetrics)")
def binary_average_precision():
    from torchmetrics.classification import BinaryAveragePrecision

    return TorchMetric(BinaryAveragePrecision, "average_precision")


class ClassMetric:
    """Accuracy or F1 of class logits: argmax over the last dimension, a single logit thresholded at zero."""

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


@lego("/metric/torchmetrics/accuracy", state=True, alias="accuracy",
            description="Accuracy of class logits (argmax) against integer labels")
def accuracy():
    return ClassMetric("accuracy", "micro")


@lego("/metric/torchmetrics/f1", state=True, alias="f1",
            description="Macro F1 of class logits (argmax) against integer labels")
def f1(average="macro"):
    return ClassMetric("f1", average)


DEFAULT_EXTRACTOR = None


def inception_features(images):
    """Inception pool features through torchmetrics (needs its FID extra); the default FID extractor."""
    from torchmetrics.image.fid import FrechetInceptionDistance

    metric = FrechetInceptionDistance(feature=2048, normalize=True)
    metric.inception.eval()
    with torch.no_grad():
        return metric.inception((images.clamp(-1.0, 1.0) * 0.5 + 0.5).float())


@lego("/lego/kalfa/pixel_features",
            description="A cheap FID feature extractor for demos and tests: images pooled to size by size and "
                        "flattened; pass it as fid's extractor param")
def pixel_features(size=4):
    def extract(images):
        return torch.nn.functional.adaptive_avg_pool2d(images, int(size)).flatten(1)

    return extract


def frechet_distance(real, fake):
    from scipy import linalg

    real = real.astype("float64")
    fake = fake.astype("float64")
    mean_real, mean_fake = real.mean(axis=0), fake.mean(axis=0)
    cov_real = numpy_cov(real)
    cov_fake = numpy_cov(fake)
    covmean = linalg.sqrtm(cov_real.dot(cov_fake))
    if not numpy_isfinite(covmean):
        offset = numpy_eye(cov_real.shape[0]) * 1e-6
        covmean = linalg.sqrtm((cov_real + offset).dot(cov_fake + offset))
    covmean = covmean.real
    difference = mean_real - mean_fake
    return float(difference.dot(difference) + numpy_trace(cov_real) + numpy_trace(cov_fake) - 2.0 * numpy_trace(covmean))


def numpy_cov(values):
    import numpy

    return numpy.cov(values, rowvar=False) if len(values) > 1 else numpy.zeros((values.shape[1], values.shape[1]))


def numpy_isfinite(values):
    import numpy

    return bool(numpy.isfinite(values).all())


def numpy_eye(count):
    import numpy

    return numpy.eye(count)


def numpy_trace(values):
    import numpy

    return float(numpy.trace(values))


class Fid:
    """Fréchet distance between the features of up to n real images and n samples of the model."""

    def __init__(self, model, latent, conditional, n, extractor=None):
        self.model = model
        self.latent = int(latent)
        self.conditional = bool(conditional)
        self.n = int(n)
        self.extractor = extractor
        self.reset()

    def reset(self):
        self.real = []
        self.fake = []
        self.seen_real = 0
        self.seen_fake = 0

    def _features(self, images):
        extractor = self.extractor or DEFAULT_EXTRACTOR or inception_features
        return extractor(images.detach().float()).detach().cpu().flatten(1).numpy()

    def update(self, models, batch, rng=None):
        images = batch["image"]
        if self.seen_real < self.n:
            take = images[:self.n - self.seen_real]
            self.real.append(self._features(take))
            self.seen_real += len(take)
        if self.seen_fake < self.n:
            count = min(len(images), self.n - self.seen_fake)
            generator = models[self.model]
            noise = torch.randn((count, self.latent), generator=rng, device=images.device) if rng is not None \
                else torch.randn((count, self.latent), device=images.device)
            with torch.no_grad():
                if self.conditional:
                    samples = generator(noise, batch["label"][:count])
                else:
                    samples = generator(noise)
            self.fake.append(self._features(samples))
            self.seen_fake += count

    def compute(self):
        import numpy

        if not self.real or not self.fake:
            return math.nan
        return frechet_distance(numpy.concatenate(self.real), numpy.concatenate(self.fake))


@lego("/metric/kalfa/fid", state=True, refs={"model": "model"}, uses=["models", "batch"],
            alias="fid", description="Fréchet inception distance of n samples of the model against n real images "
                                     "of the set; conditional samples use the batch labels")
def fid(model, latent, conditional=False, n=1000, extractor=None):
    return Fid(model, latent, conditional, n, extractor)


class Perplexity:
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


class SampleWriter:
    """Writes n samples once per pass under samples/turn_<n>: from the sampler (the generate section's call) or,
    without one, the predicts model's outputs on the batch; reports nothing to the history."""

    def __init__(self, n, sampler):
        self.n = int(n)
        self.sampler = sampler
        self.reset()

    def reset(self):
        self.done = False

    def update(self, models, predicts, rng, record, turn, batch=None, prep=None):
        from pathlib import Path

        from .eval import write_turn_samples
        from .runtime import call_model, named_outputs, parameter_names, resolve_model

        if self.done or record is None:
            return
        self.done = True
        if self.sampler is not None:
            extra = {"n": self.n} if "n" in parameter_names(self.sampler) else {}
            samples = self.sampler(models=models, prep=prep, rng=rng, **extra)
        else:
            if predicts is None or batch is None:
                raise ValueError("sample_writer needs a sampler (sampler: generate) or a predicts model and a batch")
            model = resolve_model(predicts, models)
            model.eval()
            with torch.no_grad():
                outputs = named_outputs(model, call_model(model, batch))
            samples = outputs[next(iter(outputs))][:self.n]
        write_turn_samples(samples, Path(record) / "samples", turn)

    def compute(self):
        return None


@lego("/metric/kalfa/sample_writer", state=True, alias="sample_writer",
            refs={"sampler": "generate"}, uses=["models"],
            description="A metric that writes n samples per pass under samples/turn_<n> (png and pt, or txt) from "
                        "the sampler (sampler: generate takes the generate section) or the predicts model; it "
                        "reports no value, use every and sets to pace it")
def sample_writer(n=16, sampler=None):
    return SampleWriter(n, sampler)
