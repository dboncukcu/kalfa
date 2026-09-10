import torch

from kalfa.registration import lego
from kalfa.std.objective.base import input_of


def gaussian_kl(mu, logvar):
    return -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1))


@lego("/objective/kalfa/vae", partial=True,
      refs={"encoder": "model", "decoder": "model", "recon": "criterion", "kl_schedule": "schedule"},
      alias="vae", description="VAE loss: w_rec * recon(decoder(z), x) + kl_schedule(step) * KL, z sampled from "
                               "the encoder's mu and logvar; returns loss, recon, kl and w_kl")
def vae(models, batch, encoder, decoder, recon, w_rec=1.0, kl_schedule=None, step=None, rng=None):
    enc = models[encoder]
    x = input_of(enc, batch)
    mu, logvar = enc(x)
    noise = torch.randn(mu.shape, generator=rng, device=mu.device, dtype=mu.dtype) if rng is not None \
        else torch.randn_like(mu)
    z = mu + torch.exp(0.5 * logvar) * noise
    x_hat = models[decoder](z)
    reconstruction = recon(x_hat, x)
    kl = gaussian_kl(mu, logvar)
    w_kl = float(kl_schedule(step or 0)) if kl_schedule is not None else 1.0
    return {"loss": float(w_rec) * reconstruction + w_kl * kl, "recon": reconstruction, "kl": kl, "w_kl": w_kl}
