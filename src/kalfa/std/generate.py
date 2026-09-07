"""Generation: partial legos (models, prep, rng, **params) that produce samples after training or on demand."""

import torch

from ..registration import lego


def _pick(models, name):
    if name not in models:
        raise KeyError(f"generate: {name!r} is no model; the models are {sorted(models)}")
    return models[name]


@lego("/generate/kalfa/gan_sampler", partial=True, refs={"model": "model"}, alias="gan_sampler",
            description="n samples of a generator from latent noise; conditional samples cycle through n_classes")
def gan_sampler(models, prep, rng, model, latent, n=64, conditional=False, n_classes=None):
    generator = _pick(models, model)
    generator.eval()
    device = next(iter(generator.parameters()), torch.zeros(1)).device
    noise = torch.randn((int(n), int(latent)), generator=rng, device=device) if rng is not None \
        else torch.randn((int(n), int(latent)), device=device)
    with torch.no_grad():
        if conditional:
            labels = torch.arange(int(n), device=device) % int(n_classes)
            return generator(noise, labels).cpu()
        return generator(noise).cpu()


@lego("/generate/kalfa/ddpm_sampler", partial=True, refs={"model": "model", "schedule": "schedule"},
            alias="ddpm_sampler", description="n samples by the reverse diffusion of the noise schedule from pure noise")
def ddpm_sampler(models, prep, rng, model, schedule, shape, n=64):
    from .objective import diffusion_steps, noise_schedule

    net = _pick(models, model)
    net.eval()
    device = next(iter(net.parameters()), torch.zeros(1)).device
    steps = diffusion_steps(schedule)
    betas, alphas, cumulative = noise_schedule(schedule, device)
    size = (int(n),) + tuple(int(part) for part in shape)
    x = torch.randn(size, generator=rng, device=device) if rng is not None else torch.randn(size, device=device)
    with torch.no_grad():
        for step in reversed(range(steps)):
            t = torch.full((int(n),), step, device=device, dtype=torch.long)
            predicted = net(x, t).float()
            coefficient = betas[step] / (1.0 - cumulative[step]).sqrt()
            x = (x - coefficient * predicted) / alphas[step].sqrt()
            if step > 0:
                noise = torch.randn(size, generator=rng, device=device) if rng is not None \
                    else torch.randn(size, device=device)
                x = x + betas[step].sqrt() * noise
    return x.clamp(-1.0, 1.0).cpu()


def context_of(net, context):
    """The window of tokens the model sees: ``context`` when given, else the seq_len a module of the model declares."""
    if context is not None:
        return int(context)
    for module in net.modules():
        if hasattr(module, "seq_len"):
            return int(module.seq_len)
    return None


@lego("/generate/kalfa/lm_sampler", partial=True, refs={"model": "model"}, alias="lm_sampler",
            description="Autoregressive text from a prompt with the record's tokenizer; temperature scales the "
                        "logits, the window is context or the model's seq_len; the model's last layer has one "
                        "logit per vocabulary entry (vocab_size)")
def lm_sampler(models, prep, rng, model, prompt, max_new_tokens=100, temperature=1.0, context=None):
    net = _pick(models, model)
    net.eval()
    tokenizer = prep.tokenizer() if prep is not None else None
    if tokenizer is None:
        raise ValueError("lm_sampler needs a fitted tokenizer among the preprocessors")
    device = next(iter(net.parameters()), torch.zeros(1)).device
    ids = torch.as_tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).reshape(1, -1)
    limit = context_of(net, context)
    with torch.no_grad():
        for _ in range(int(max_new_tokens)):
            window = ids[:, -int(limit):] if limit else ids
            logits = net(window)[:, -1, :].float() / max(float(temperature), 1e-6)
            probabilities = torch.softmax(logits, dim=-1)
            pick = torch.multinomial(probabilities, 1, generator=rng) if rng is not None \
                else torch.multinomial(probabilities, 1)
            ids = torch.cat([ids, pick], dim=1)
    return tokenizer.decode(ids[0].cpu().numpy())
