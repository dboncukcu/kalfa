from pathlib import Path

import torch

from kalfa.std.common.generation import write_turn_samples
from kalfa.std.common.runtime import call_model, named_outputs, parameter_names, resolve_model
from kalfa.std.metric.base import Metric


class SampleWriter(Metric):
    def __init__(self, n=16, sampler=None):
        self.n = int(n)
        self.sampler = sampler
        self.reset()

    def reset(self):
        self.done = False

    def update(self, models, predicts, rng, record, turn, batch=None, prep=None):
        if self.done or record is None:
            return
        self.done = True
        if self.sampler is not None:
            extra = {"n": self.n} if "n" in parameter_names(self.sampler) else {}
            samples = self.sampler(models=models, prep=prep, rng=rng, **extra)
        else:
            if predicts is None or batch is None:
                raise ValueError("sample_writer needs a sampler (sampler: generate) or a predicts model and a batch")
            model = resolve_model(predicts, models)
            model.eval()
            with torch.no_grad():
                outputs = named_outputs(model, call_model(model, batch))
            samples = outputs[next(iter(outputs))][:self.n]
        write_turn_samples(samples, Path(record) / "samples", turn)

    def compute(self):
        return None
