import torch

from kalfa.std.common.diffusion import diffusion_steps, noise_schedule
from kalfa.std.objective.base import input_of


def ddpm(models, batch, model, schedule, rng=None):
    net = models[model]
    x0 = input_of(net, batch)
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
