import torch


def diffusion_steps(schedule):
    """The number of steps a noise schedule spans: the steps param the partial carries."""
    keywords = getattr(schedule, "keywords", None) or {}
    steps = keywords.get("steps")
    if steps is None:
        raise ValueError("the noise schedule must be a schedule lego with a steps param (linear_betas)")
    return int(steps)


noise_cache = {}


def noise_schedule(schedule, device):
    """betas, alphas and cumulative alphas of a schedule as tensors on the device, cached per schedule."""
    key = (getattr(schedule, "func", schedule), tuple(sorted((getattr(schedule, "keywords", None) or {}).items())),
           str(device))
    if key not in noise_cache:
        steps = diffusion_steps(schedule)
        betas = torch.tensor([float(schedule(step)) for step in range(steps)], dtype=torch.float32, device=device)
        alphas = 1.0 - betas
        noise_cache[key] = (betas, alphas, torch.cumprod(alphas, dim=0))
    return noise_cache[key]
