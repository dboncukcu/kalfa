from kalfa.registration import pack

lego = pack(__name__)


lego("/turn/kalfa/alternating", "alternating:alternating", alias=["alternating", "supervised"],
     returns=["models", "optimizers", "emas", "counters", "stream", "metrics"],
     mutates=["models", "optimizers", "emas", "counters"], bus=["device", "prep", "record", "monitor"],
     extras=["amp", "grad_clip", "accumulate"],
     description="One turn: every step each optimizer in order minimizes its loss for its steps; a turn is an epoch, "
                 "or K steps with a stream that lives across turns; losses and metrics are the running means of the "
                 "pass; every update writes a line to steps.jsonl (the loss, the learning rate and, under grad_clip, "
                 "the gradient norm of each optimizer) and reaches the monitor")
