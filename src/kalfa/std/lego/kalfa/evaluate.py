import torch

from kalfa.registration import lego
from kalfa.std.common.device import Device
from kalfa.std.common.log import clock, logger_for, since
from kalfa.std.common.runtime import (Context, Pass, active_entries, collect_results, observe_all, resolve_entries,
                                      set_modes)


logger = logger_for("training.eval")


@lego("/lego/kalfa/evaluate", returns="metrics", bus=["device", "prep", "record"],
      description="Losses (model scale) and metrics (original scale, through prep) of one set under no_grad; "
                  "an empty set gives an empty mapping; record reaches metrics that write files")
def evaluate(models, emas, composites, counters, effects, loader, set, losses, metrics, losses_keys, metrics_keys,
             predicts, device=None, prep=None, record=None):
    if loader is None or loader.dataset.size() == 0:
        return {}
    turn = int(counters.get("turn", 0))
    trackers = [entry.tracker(name, keys) for name, entry, keys in active_entries(losses, losses_keys, set, turn)]
    trackers += [entry.tracker(name, keys, rescale=True)
                 for name, entry, keys in active_entries(metrics, metrics_keys, set, turn)]
    if not trackers:
        return {}
    set_modes(models, train=False, composites=composites)
    resolve_entries(metrics, loader=loader)
    device = device or Device.cpu()
    scope = Pass(models, composites, emas, predicts, list(loader.dataset.targets), epoch=turn, device=device,
                 rng=device.generator(), prep=prep, set_name=set, losses=losses, losses_keys=dict(losses_keys or {}),
                 record=record)
    seen = 0
    started = clock()
    with torch.no_grad():
        for batch in loader:
            seen += 1
            observe_all(trackers, Context(device.move(batch), scope, counters.get("global_step", 0)))
    if not seen:
        return {}
    logger.debug(f"{set}: {seen} batches ({since(started)})")
    return collect_results(trackers)
