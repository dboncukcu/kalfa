"""The plugin of the alad example: the two ALAD objectives, registered with their facts."""

import kalfa
import torch

MODELS = ["encoder", "generator", "dxz", "dxx", "dzz"]


def _first(value):
    return value[0] if isinstance(value, (tuple, list)) else value


def _feature(value):
    return value[1] if isinstance(value, (tuple, list)) else value


def _latent(batch, latent_dim, rng):
    x = batch["x"]
    return torch.randn(x.shape[0], int(latent_dim), device=x.device, generator=rng)


def _ones(logit):
    return torch.ones_like(logit)


def _zeros(logit):
    return torch.zeros_like(logit)


@kalfa.lego("/objective/myexample/alad_discriminator", partial=True, refs={"criterion": "criterion"},
            needs_models=MODELS, description="ALAD discriminator loss over dxz, dxx and dzz with detached generator "
                                             "and encoder outputs")
def alad_discriminator(models, batch, criterion, latent_dim, rng=None):
    x = batch["x"]
    z = _latent(batch, latent_dim, rng)
    with torch.no_grad():
        z_hat = models["encoder"](x)
        x_gen = models["generator"](z)
        x_rec = models["generator"](z_hat)
        z_rec = models["encoder"](x_gen)
    real_xz = _first(models["dxz"](x, z_hat))
    fake_xz = _first(models["dxz"](x_gen, z))
    real_xx = _first(models["dxx"](x, x))
    fake_xx = _first(models["dxx"](x, x_rec))
    real_zz = _first(models["dzz"](z, z))
    fake_zz = _first(models["dzz"](z, z_rec))
    return (criterion(real_xz, _ones(real_xz)) + criterion(fake_xz, _zeros(fake_xz))
            + criterion(real_xx, _ones(real_xx)) + criterion(fake_xx, _zeros(fake_xx))
            + criterion(real_zz, _ones(real_zz)) + criterion(fake_zz, _zeros(fake_zz)))


@kalfa.lego("/objective/myexample/alad_generator", partial=True, refs={"criterion": "criterion"},
            needs_models=MODELS, description="ALAD encoder and generator loss with flipped labels and optional "
                                             "feature matching on dxx")
def alad_generator(models, batch, criterion, latent_dim, feature_matching=0.0, rng=None):
    x = batch["x"]
    z = _latent(batch, latent_dim, rng)
    z_hat = models["encoder"](x)
    x_gen = models["generator"](z)
    x_rec = models["generator"](z_hat)
    z_rec = models["encoder"](x_gen)
    real_xz = _first(models["dxz"](x, z_hat))
    fake_xz = _first(models["dxz"](x_gen, z))
    rec_xx = models["dxx"](x, x_rec)
    ref_xx = models["dxx"](x, x)
    fake_zz = _first(models["dzz"](z, z_rec))
    real_zz = _first(models["dzz"](z, z))
    loss = (criterion(real_xz, _zeros(real_xz)) + criterion(fake_xz, _ones(fake_xz))
            + criterion(_first(rec_xx), _ones(_first(rec_xx))) + criterion(_first(ref_xx), _zeros(_first(ref_xx)))
            + criterion(fake_zz, _ones(fake_zz)) + criterion(real_zz, _zeros(real_zz)))
    if feature_matching:
        loss = loss + float(feature_matching) * (_feature(rec_xx) - _feature(ref_xx)).abs().mean()
    return loss
