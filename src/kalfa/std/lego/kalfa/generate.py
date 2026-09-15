from pathlib import Path

from kalfa.std.common.generation import write_samples
from kalfa.std.common.log import logger_for
from kalfa.std.common.device import Device


logger_after = logger_for("after")


def generate(models, composites, prep, generate, record=None):
    if generate is None:
        return None
    samples = generate(models={**dict(composites or {}), **dict(models)}, prep=prep, rng=Device.cpu().generator())
    if record is not None and samples is not None:
        write_samples(samples, Path(record) / "samples")
        logger_after.info(f"samples written under {Path(record) / 'samples'}")
    return None
