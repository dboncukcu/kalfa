from pathlib import Path

from kalfa.registration import lego
from kalfa.std.checkpoint.base import payload, save
from kalfa.std.common.log import logger_for, number


logger_checkpoint = logger_for("training.ckpt")


def checkpoint_line(policy, tags, metrics):
    extra = [tag for tag in tags if tag != "last"]
    if extra:
        logger_checkpoint.info("wrote " + ", ".join(f"{tag}.pt" for tag in tags))
        return
    monitor = policy.monitor
    if monitor is None:
        logger_checkpoint.debug("wrote last.pt")
        return
    value = (metrics or {}).get(monitor)
    if value is None:
        logger_checkpoint.debug(f"wrote last.pt; {monitor} is not in this turn's metrics")
    else:
        logger_checkpoint.debug(f"wrote last.pt; {monitor} {number(value)} is no better than {number(policy.best)}")


def strip_suffix(state, suffix="_next"):
    return {key[:-len(suffix)] if key.endswith(suffix) else key: value for key, value in state.items()}


@lego("/lego/kalfa/checkpoint", returns=None, bus=["metrics", "record"],
      description="Write the checkpoint files the policy asks for; nothing without a policy")
def checkpoint(state, policy, metrics=None, record=None):
    if not policy or record is None:
        return None
    parts = strip_suffix(state)
    tags = policy.tags(metrics)
    data = payload(parts.get("models"), parts.get("optimizers"), parts.get("emas"), parts.get("counters"),
                   parts.get("rules"), policy.state())
    for tag in tags:
        save(Path(record) / "checkpoints" / f"{tag}.pt", data)
    checkpoint_line(policy, tags, metrics)
    return None
