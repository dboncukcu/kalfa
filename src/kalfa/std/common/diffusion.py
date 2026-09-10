import torch


def diffusion_steps(schedule):
    keywords = getattr(schedule, "keywords", None) or {}
    steps = keywords.get("steps")
    if steps is None:
        raise ValueError("the noise schedule must be a schedule lego with a steps param (linear_betas)")
    return int(steps)


def noise_schedule(schedule, device):
    steps = diffusion_steps(schedule)
    betas = torch.tensor([float(schedule(step)) for step in range(steps)], dtype=torch.float32, device=device)
    alphas = 1.0 - betas
    return betas, alphas, torch.cumprod(alphas, dim=0)
