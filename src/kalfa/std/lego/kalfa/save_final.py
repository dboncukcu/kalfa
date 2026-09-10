from pathlib import Path

from kalfa.registration import lego
from kalfa.std.checkpoint.base import payload, save
from kalfa.std.common.log import logger_for


logger_after = logger_for("after")


@lego("/lego/kalfa/save_final", returns=None, bus=["record"],
      description="Write final/state.pt with the full state once training ends")
def save_final(models, optimizers, emas, counters, rules, record=None):
    if record is None:
        return None
    save(Path(record) / "final" / "state.pt", payload(models, optimizers, emas, counters, rules))
    logger_after.debug("final/state.pt written")
    return None
