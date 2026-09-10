from kalfa.registration import lego


@lego("/schedule/kalfa/linear_betas", partial=True, alias="linear_betas",
      description="The diffusion noise schedule: beta rises linearly from start to end over steps")
def linear_betas(step, steps, start=1e-4, end=0.02):
    if steps <= 1:
        return float(end)
    return float(start) + (float(end) - float(start)) * (float(step) / float(steps - 1))
