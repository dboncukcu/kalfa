"""Schedules: partial legos mapping a step to a value."""

import math

from ..registration import lego


@lego("/schedule/kalfa/linear_warmup", partial=True, alias="linear_warmup",
            description="Linear ramp from start to end over steps updates, then end")
def linear_warmup(step, start, end, steps):
    if steps <= 0:
        return float(end)
    fraction = min(max(float(step) / float(steps), 0.0), 1.0)
    return float(start) + (float(end) - float(start)) * fraction


@lego("/schedule/kalfa/step_decay", partial=True, alias="step_decay",
            description="Multiply by gamma every step_size updates; the value is the factor")
def step_decay(step, step_size, gamma):
    return float(gamma) ** (int(step) // int(step_size))


@lego("/schedule/kalfa/warmup_cosine", partial=True, alias="warmup_cosine",
            description="Linear warmup to one over warmup updates, then a cosine decay to zero at total")
def warmup_cosine(step, warmup, total):
    step = float(step)
    if warmup and step < warmup:
        return step / float(warmup)
    if total <= warmup:
        return 0.0
    progress = min((step - warmup) / float(total - warmup), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@lego("/schedule/kalfa/linear_betas", partial=True, alias="linear_betas",
            description="The diffusion noise schedule: beta rises linearly from start to end over steps")
def linear_betas(step, steps, start=1e-4, end=0.02):
    if steps <= 1:
        return float(end)
    return float(start) + (float(end) - float(start)) * (float(step) / float(steps - 1))
