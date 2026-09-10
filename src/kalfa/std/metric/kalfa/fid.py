import math

import numpy
import torch

from kalfa.registration import lego
from kalfa.std.metric.base import Metric


def inception_features(images):
    from torchmetrics.image.fid import FrechetInceptionDistance

    metric = FrechetInceptionDistance(feature=2048, normalize=True)
    metric.inception.eval()
    with torch.no_grad():
        return metric.inception((images.clamp(-1.0, 1.0) * 0.5 + 0.5).float())


def frechet_distance(real, fake):
    from scipy import linalg

    real = real.astype("float64")
    fake = fake.astype("float64")
    mean_real, mean_fake = real.mean(axis=0), fake.mean(axis=0)
    cov_real = covariance(real)
    cov_fake = covariance(fake)
    covmean = linalg.sqrtm(cov_real.dot(cov_fake))
    if not numpy.isfinite(covmean).all():
        offset = numpy.eye(cov_real.shape[0]) * 1e-6
        covmean = linalg.sqrtm((cov_real + offset).dot(cov_fake + offset))
    covmean = covmean.real
    difference = mean_real - mean_fake
    return float(difference.dot(difference) + numpy.trace(cov_real) + numpy.trace(cov_fake)
                 - 2.0 * numpy.trace(covmean))


def covariance(values):
    return numpy.cov(values, rowvar=False) if len(values) > 1 else numpy.zeros((values.shape[1], values.shape[1]))


@lego("/metric/kalfa/fid", state=True, refs={"model": "model"}, uses=["models", "batch"],
      alias="fid", description="Fréchet inception distance of n samples of the model against n real images "
                               "of the set; conditional samples use the batch labels")
class Fid(Metric):
    def __init__(self, model, latent, conditional=False, n=1000, extractor=None):
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

    def features_of(self, images):
        extractor = self.extractor or inception_features
        return extractor(images.detach().float()).detach().cpu().flatten(1).numpy()

    def update(self, models, batch, rng=None):
        images = batch["image"]
        if self.seen_real < self.n:
            take = images[:self.n - self.seen_real]
            self.real.append(self.features_of(take))
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
            self.fake.append(self.features_of(samples))
            self.seen_fake += count

    def compute(self):
        if not self.real or not self.fake:
            return math.nan
        return frechet_distance(numpy.concatenate(self.real), numpy.concatenate(self.fake))
