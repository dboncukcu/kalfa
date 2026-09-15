import math


def linear_warmup(step, start, end, steps):
    if steps <= 0:
        return float(end)
    fraction = min(max(float(step) / float(steps), 0.0), 1.0)
    return float(start) + (float(end) - float(start)) * fraction


def warmup_cosine(step, warmup, total):
    step = float(step)
    if warmup and step < warmup:
        return step / float(warmup)
    if total <= warmup:
        return 0.0
    progress = min((step - warmup) / float(total - warmup), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def step_decay(step, step_size, gamma):
    return float(gamma) ** (int(step) // int(step_size))


def linear_betas(step, steps, start=1e-4, end=0.02):
    if steps <= 1:
        return float(end)
    return float(start) + (float(end) - float(start)) * (float(step) / float(steps - 1))
