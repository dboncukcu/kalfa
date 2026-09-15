from kalfa.registration import pack

lego = pack(__name__)


lego("/schedule/kalfa/linear_warmup", "schedules:linear_warmup", partial=True, alias="linear_warmup",
     description="Linear ramp from start to end over steps updates, then end")
lego("/schedule/kalfa/warmup_cosine", "schedules:warmup_cosine", partial=True, alias="warmup_cosine",
     description="Linear warmup to one over warmup updates, then a cosine decay to zero at total")
lego("/schedule/kalfa/step_decay", "schedules:step_decay", partial=True, alias="step_decay",
     description="Multiply by gamma every step_size updates; the value is the factor")
lego("/schedule/kalfa/linear_betas", "schedules:linear_betas", partial=True, alias="linear_betas",
     description="The diffusion noise schedule: beta rises linearly from start to end over steps")
