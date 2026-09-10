import torch

from kalfa.registration import lego
from kalfa.std.metric.base import Metric
from pathlib import Path
from kalfa.std.common.generation import write_turn_samples
from kalfa.std.common.runtime import call_model, named_outputs, parameter_names, resolve_model


@lego("/metric/kalfa/sample_writer", state=True, alias="sample_writer",
      refs={"sampler": "generate"}, uses=["models"],
      description="A metric that writes n samples per pass under samples/turn_<n> (png and pt, or txt) from "
                  "the sampler (sampler: generate takes the generate section) or the predicts model; it "
                  "reports no value, use every and sets to pace it")
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
