from kalfa.registration import lego


@lego("/schedule/kalfa/linear_warmup", partial=True, alias="linear_warmup",
      description="Linear ramp from start to end over steps updates, then end")
def linear_warmup(step, start, end, steps):
    if steps <= 0:
        return float(end)
    fraction = min(max(float(step) / float(steps), 0.0), 1.0)
    return float(start) + (float(end) - float(start)) * fraction
