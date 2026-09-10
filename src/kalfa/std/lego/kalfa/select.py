import copy
from pathlib import Path

from kalfa.registration import lego
from kalfa.std.checkpoint.base import load
from kalfa.std.common.log import logger_for


logger_after = logger_for("after")


@lego("/lego/kalfa/select", returns="selected", bus=["record"],
      description="The report models: copies loaded from best.pt, or the final state for last")
def select(models, emas, which, record=None):
    copies = copy.deepcopy(dict(models))
    ema_copies = copy.deepcopy(dict(emas or {}))
    if which == "best":
        path = Path(record or ".") / "checkpoints" / "best.pt"
        if not path.exists():
            raise FileNotFoundError(f"report: best needs {path}, but the best checkpoint was never written")
        data = load(path)
        logger_after.info(f"report best: the checkpoint of turn {data.get('turn')}")
        for name, state in data.get("models", {}).items():
            if name in copies:
                copies[name].load_state_dict(state)
        for name, state in data.get("emas", {}).items():
            if name in ema_copies:
                ema_copies[name].load_state_dict(state)
    elif which != "last":
        raise ValueError(f"report must be best or last, got {which!r}")
    else:
        logger_after.info("report last: the models as training left them")
    selected = dict(copies)
    for name, ema in ema_copies.items():
        selected[f"{name}.ema"] = ema
    for model in selected.values():
        model.eval()
    return selected
