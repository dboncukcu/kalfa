import torch

from kalfa.registration import lego
from kalfa.std.generate.base import pick_model
from kalfa.std.common.diffusion import diffusion_steps, noise_schedule


@lego("/generate/kalfa/ddpm_sampler", partial=True, refs={"model": "model", "schedule": "schedule"},
      alias="ddpm_sampler", description="n samples by the reverse diffusion of the noise schedule from pure noise")
def ddpm_sampler(models, prep, rng, model, schedule, shape, n=64):
    net = pick_model(models, model)
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
