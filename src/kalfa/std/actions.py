import torch

from kalfa.core.aliases import register
from kalfa.core.errors import BuildError, RunError
from kalfa.core.protocols import Action
from kalfa.std.samples import save_samples


class WriteSamples(Action):
    requires_generate = True

    def __init__(self, sampler=None, n=16):
        self.sampler = sampler
        self.n = int(n)

    def bind(self, ctx):
        if self.sampler is None and getattr(ctx, "generate", None) is None:
            raise BuildError(f"{self.path or 'write_samples'}: write_samples needs a sampler: write sampler: "
                             "{use: gan_sampler, ...} or a generate section")

    def __call__(self, ctx):
        sampler = self.sampler if self.sampler is not None else getattr(ctx, "generate", None)
        if sampler is None:
            raise RunError(f"{self.path or 'write_samples'} needs a sampler: write sampler: <a sampler call> or a "
                           "generate section")
        with torch.no_grad():
            samples = sampler.sample(ctx, self.n)
        if getattr(ctx, "record", None) is None:
            return []
        return save_samples(samples, ctx.record.path("samples"), f"turn_{ctx.turn}")


register("write_samples", "kalfa.std.actions.WriteSamples", slot="action",
         doc="draws n samples from its sampler (or the run's generate) and writes samples/turn_<k>.pt, a png grid "
             "for images or a txt for text")
