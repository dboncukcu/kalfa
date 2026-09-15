import torch

from kalfa.std.objective.base import call_with, input_of, labels_of, latent_noise


def wgan_g(models, batch, generator, critic, latent, conditional=False, rng=None):
    real = input_of(models[critic], batch)
    labels = labels_of(batch, conditional)
    noise = latent_noise(real.shape[0], latent, real, rng)
    fake = call_with(models[generator], noise, labels)
    return -call_with(models[critic], fake, labels).mean()


def wgan_gp_d(models, batch, generator, critic, latent, gp_weight=10.0, conditional=False, rng=None, scaler=None):
    real = input_of(models[critic], batch)
    labels = labels_of(batch, conditional)
    noise = latent_noise(real.shape[0], latent, real, rng)
    with torch.no_grad():
        fake = call_with(models[generator], noise, labels)
    score_real = call_with(models[critic], real, labels)
    score_fake = call_with(models[critic], fake, labels)
    shape = (real.shape[0],) + (1,) * (real.ndim - 1)
    mix = torch.rand(shape, generator=rng, device=real.device, dtype=real.dtype) if rng is not None \
        else torch.rand(shape, device=real.device, dtype=real.dtype)
    between = (mix * real + (1.0 - mix) * fake).requires_grad_(True)
    score_between = call_with(models[critic], between, labels)
    gradients = torch.autograd.grad(score_between.sum(), between, create_graph=True)[0]
    penalty = ((gradients.flatten(1).norm(dim=1) - 1.0) ** 2).mean()
    return score_fake.mean() - score_real.mean() + float(gp_weight) * penalty
