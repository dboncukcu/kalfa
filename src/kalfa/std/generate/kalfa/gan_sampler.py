import torch

from kalfa.registration import lego
from kalfa.std.generate.base import pick_model


@lego("/generate/kalfa/gan_sampler", partial=True, refs={"model": "model"}, alias="gan_sampler",
      description="n samples of a generator from latent noise; conditional samples cycle through n_classes")
def gan_sampler(models, prep, rng, model, latent, n=64, conditional=False, n_classes=None):
    generator = pick_model(models, model)
    generator.eval()
    device = next(iter(generator.parameters()), torch.zeros(1)).device
    noise = torch.randn((int(n), int(latent)), generator=rng, device=device) if rng is not None \
        else torch.randn((int(n), int(latent)), device=device)
    with torch.no_grad():
        if conditional:
            labels = torch.arange(int(n), device=device) % int(n_classes)
            return generator(noise, labels).cpu()
        return generator(noise).cpu()
