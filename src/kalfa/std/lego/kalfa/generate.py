from pathlib import Path

from kalfa.registration import lego
from kalfa.std.common.generation import write_samples
from kalfa.std.common.log import logger_for
from kalfa.std.common.runtime import turn_generator


logger_after = logger_for("after")


@lego("/lego/kalfa/generate", returns=None, bus=["record"],
      description="Run the generate lego with the report models; nothing without a generate section")
def generate(models, composites, prep, generate, record=None):
    if generate is None:
        return None
    samples = generate(models={**dict(composites or {}), **dict(models)}, prep=prep, rng=turn_generator())
    if record is not None and samples is not None:
        write_samples(samples, Path(record) / "samples")
        logger_after.info(f"samples written under {Path(record) / 'samples'}")
    return None
