import math

from kalfa.registration import lego


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
