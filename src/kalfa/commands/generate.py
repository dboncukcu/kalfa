from dataclasses import dataclass
from pathlib import Path

import torch

from kalfa.commands.common import loaded_which
from kalfa.core.errors import KalfaError
from kalfa.core.run import open_record
from kalfa.std.samples import save_samples


@dataclass
class Generated:
    record: str
    paths: list
    samples: object
    which: str


def generate_record(record, n=None, which=None, device=None, out=None):
    ctx = open_record(record, which=which, device=device, data=False)
    source = Path(record)
    sampler = ctx.generate
    if sampler is None:
        raise KalfaError(f"{record} has no generate section")
    which = loaded_which(ctx, source, which)
    count = n if n is not None else getattr(sampler, "n", None)
    with torch.no_grad():
        samples = sampler.sample(ctx, count)
    written = save_samples(samples, out or source / "samples", f"generate_{which}")
    return Generated(record=str(record), paths=[str(path) for path in written], samples=samples, which=which)
