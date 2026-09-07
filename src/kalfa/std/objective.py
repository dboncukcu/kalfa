"""Objectives: (models, batch, **params) legos that wire several models inside the loss."""

import torch

from ..registration import lego


def _input_of(model, batch):
    wires = list(getattr(model, "inputs", []))
    if not wires or wires[0] not in batch:
        raise KeyError(f"the objective needs the model's first input wire in the batch; wires {wires}, "
                       f"batch {sorted(batch)}")
    return batch[wires[0]]


def _gaussian_kl(mu, logvar):
    return -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1))


@lego("/objective/kalfa/vae", partial=True,
            refs={"encoder": "model", "decoder": "model", "recon": "criterion", "kl_schedule": "schedule"},
            alias="vae", description="VAE loss: w_rec * recon(decoder(z), x) + kl_schedule(step) * KL, z sampled from "
                                     "the encoder's mu and logvar; returns loss, recon, kl and w_kl")
def vae(models, batch, encoder, decoder, recon, w_rec=1.0, kl_schedule=None, step=None, rng=None):
    enc = models[encoder]
    x = _input_of(enc, batch)
    mu, logvar = enc(x)
    noise = torch.randn(mu.shape, generator=rng, device=mu.device, dtype=mu.dtype) if rng is not None \
        else torch.randn_like(mu)
    z = mu + torch.exp(0.5 * logvar) * noise
    x_hat = models[decoder](z)
    reconstruction = recon(x_hat, x)
    kl = _gaussian_kl(mu, logvar)
    w_kl = float(kl_schedule(step or 0)) if kl_schedule is not None else 1.0
    return {"loss": float(w_rec) * reconstruction + w_kl * kl, "recon": reconstruction, "kl": kl, "w_kl": w_kl}


def _target_of(batch, model, target=None):
    if target is not None:
        return batch[target]
    wires = set(getattr(model, "inputs", []))
    rest = [name for name in batch if name not in wires]
    if len(rest) != 1:
        raise ValueError(f"the objective cannot tell the target field among {rest}; write target")
    return batch[rest[0]]


@lego("/objective/kalfa/distill", partial=True, refs={"student": "model", "teacher": "model"},
            alias="distill", description="Knowledge distillation: alpha * KL(teacher || student) at temperature T "
                                         "(times T squared) plus (1 - alpha) * cross entropy of the student; "
                                         "returns loss, ce and kl")
def distill(models, batch, student, teacher, temperature=1.0, alpha=0.5, target=None):
    learner = models[student]
    x = _input_of(learner, batch)
    labels = _target_of(batch, learner, target).reshape(-1).long()
    logits = learner(x)
    with torch.no_grad():
        guide = models[teacher](x)
    scale = float(temperature)
    log_soft = torch.log_softmax(logits / scale, dim=-1)
    soft = torch.softmax(guide / scale, dim=-1)
    kl = torch.nn.functional.kl_div(log_soft, soft, reduction="batchmean") * scale * scale
    ce = torch.nn.functional.cross_entropy(logits, labels)
    return {"loss": float(alpha) * kl + (1.0 - float(alpha)) * ce, "ce": ce, "kl": kl}


def _latent_noise(count, latent, like, rng):
    return torch.randn((count, int(latent)), generator=rng, device=like.device, dtype=like.dtype) if rng is not None \
        else torch.randn((count, int(latent)), device=like.device, dtype=like.dtype)


def _labels(batch, conditional):
    if not conditional:
        return None
    if "label" not in batch:
        raise KeyError("a conditional GAN reads the label field of the batch")
    return batch["label"]


def _call(model, *parts):
    return model(*[part for part in parts if part is not None])


@lego("/objective/kalfa/wgan_gp_d", partial=True, needs_grad=True,
            refs={"generator": "model", "critic": "model"}, alias="wgan_gp_d",
            description="WGAN critic loss with a gradient penalty on interpolates; the label conditions both models")
def wgan_gp_d(models, batch, generator, critic, latent, gp_weight=10.0, conditional=False, rng=None, scaler=None):
    real = _input_of(models[critic], batch)
    labels = _labels(batch, conditional)
    noise = _latent_noise(real.shape[0], latent, real, rng)
    with torch.no_grad():
        fake = _call(models[generator], noise, labels)
    score_real = _call(models[critic], real, labels)
    score_fake = _call(models[critic], fake, labels)
    shape = (real.shape[0],) + (1,) * (real.ndim - 1)
    mix = torch.rand(shape, generator=rng, device=real.device, dtype=real.dtype) if rng is not None \
        else torch.rand(shape, device=real.device, dtype=real.dtype)
    between = (mix * real + (1.0 - mix) * fake).requires_grad_(True)
    score_between = _call(models[critic], between, labels)
    gradients = torch.autograd.grad(score_between.sum(), between, create_graph=True)[0]
    penalty = ((gradients.flatten(1).norm(dim=1) - 1.0) ** 2).mean()
    return score_fake.mean() - score_real.mean() + float(gp_weight) * penalty


@lego("/objective/kalfa/wgan_g", partial=True, refs={"generator": "model", "critic": "model"},
            alias="wgan_g", description="WGAN generator loss: minus the critic's mean score of generated samples")
def wgan_g(models, batch, generator, critic, latent, conditional=False, rng=None):
    real = _input_of(models[critic], batch)
    labels = _labels(batch, conditional)
    noise = _latent_noise(real.shape[0], latent, real, rng)
    fake = _call(models[generator], noise, labels)
    return -_call(models[critic], fake, labels).mean()


def diffusion_steps(schedule):
    """The number of steps a noise schedule spans: the steps param the partial carries."""
    keywords = getattr(schedule, "keywords", None) or {}
    steps = keywords.get("steps")
    if steps is None:
        raise ValueError("the noise schedule must be a schedule lego with a steps param (linear_betas)")
    return int(steps)


_noise_cache = {}


def noise_schedule(schedule, device):
    """betas, alphas and cumulative alphas of a schedule as tensors on the device, cached per schedule."""
    key = (getattr(schedule, "func", schedule), tuple(sorted((getattr(schedule, "keywords", None) or {}).items())),
           str(device))
    if key not in _noise_cache:
        steps = diffusion_steps(schedule)
        betas = torch.tensor([float(schedule(step)) for step in range(steps)], dtype=torch.float32, device=device)
        alphas = 1.0 - betas
        _noise_cache[key] = (betas, alphas, torch.cumprod(alphas, dim=0))
    return _noise_cache[key]


@lego("/objective/kalfa/ddpm", partial=True, refs={"model": "model", "schedule": "schedule"},
            alias="ddpm", description="DDPM noise prediction loss: a random time step and noise per sample (rng), "
                                      "the model predicts the noise of the noised input")
def ddpm(models, batch, model, schedule, rng=None):
    net = models[model]
    x0 = _input_of(net, batch)
    steps = diffusion_steps(schedule)
    _, _, cumulative = noise_schedule(schedule, x0.device)
    count = x0.shape[0]
    t = torch.randint(0, steps, (count,), generator=rng, device=x0.device) if rng is not None \
        else torch.randint(0, steps, (count,), device=x0.device)
    noise = torch.randn(x0.shape, generator=rng, device=x0.device, dtype=x0.dtype) if rng is not None \
        else torch.randn_like(x0)
    weight = cumulative[t].to(x0.dtype).reshape((count,) + (1,) * (x0.ndim - 1))
    noised = weight.sqrt() * x0 + (1.0 - weight).sqrt() * noise
    predicted = net(noised, t)
    return ((predicted.float() - noise.float()) ** 2).mean()


@lego("/objective/kalfa/ntxent", partial=True, refs={"model": "model"}, alias="ntxent",
            description="NT-Xent contrastive loss over the two views of every image in the batch")
def ntxent(models, batch, model, temperature=0.5):
    net = models[model]
    views = _input_of(net, batch)
    if views.ndim < 5:
        raise ValueError("ntxent needs two views per image: put two_views in the field's chain (sets: [train])")
    count = views.shape[0]
    flat = views.flatten(0, 1)
    z = torch.nn.functional.normalize(net(flat).float(), dim=1)
    similarity = z @ z.t() / float(temperature)
    similarity = similarity.masked_fill(torch.eye(2 * count, device=z.device, dtype=torch.bool), float("-inf"))
    positions = torch.arange(2 * count, device=z.device)
    partner = positions ^ 1
    return torch.nn.functional.cross_entropy(similarity, partner)


@lego("/objective/kalfa/weighted_sum", partial=True, refs={"terms": "loss"},
            alias="weighted_sum",
            description="The weighted sum of other losses definitions on the same batch: terms maps a losses "
                        "name to its weight; returns loss and every term")
def weighted_sum(models, batch, terms, losses):
    if not isinstance(terms, dict) or not terms:
        raise ValueError("weighted_sum needs terms: a mapping of losses names to weights")
    out = {}
    total = None
    for name, weight in terms.items():
        value = losses[name]
        scalar = value["loss"] if isinstance(value, dict) else value
        out[name] = scalar
        total = float(weight) * scalar if total is None else total + float(weight) * scalar
    out["loss"] = total
    return out
