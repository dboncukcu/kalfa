from kalfa.registration import lego


@lego("/schedule/kalfa/step_decay", partial=True, alias="step_decay",
      description="Multiply by gamma every step_size updates; the value is the factor")
def step_decay(step, step_size, gamma):
    return float(gamma) ** (int(step) // int(step_size))
