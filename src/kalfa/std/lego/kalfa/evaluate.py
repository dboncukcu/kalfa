import torch

from kalfa.registration import lego
from kalfa.std.common.log import clock, logger_for, since
from kalfa.std.common.runtime import (
    Context,
    active_entries,
    collect_results,
    observe_all,
    set_modes,
    to_device,
    tracker_for,
    turn_generator,
)
from kalfa.std.feed.base import sized


logger_eval = logger_for("training.eval")


@lego("/lego/kalfa/evaluate", returns="metrics", bus=["device", "prep", "record"],
      description="Losses (model scale) and metrics (original scale, through prep) of one set under no_grad; "
                  "an empty set gives an empty mapping; record reaches metrics that write files")
def evaluate(models, emas, composites, counters, effects, loader, set, losses, metrics, losses_keys, metrics_keys,
             predicts, device=None, prep=None, record=None):
    if loader is None or sized(loader.dataset) == 0:
        return {}
    turn = int(counters.get("turn", 0))
    trackers = [tracker_for(name, entry, keys) for name, entry, keys in active_entries(losses, losses_keys, set, turn)]
    trackers += [tracker_for(name, entry, keys, rescale=True)
                 for name, entry, keys in active_entries(metrics, metrics_keys, set, turn)]
    if not trackers:
        return {}
    targets = list(getattr(loader.dataset, "targets", []))
    set_modes(models, train=False, composites=composites)
    rng = turn_generator(device)
    seen = 0
    started = clock()
    with torch.no_grad():
        for batch in loader:
            seen += 1
            context = Context(to_device(batch, device), models, composites, emas, predicts, targets,
                              step=counters.get("global_step", 0), epoch=turn, rng=rng, prep=prep, set_name=set,
                              losses=losses, losses_keys=losses_keys, record=record)
            observe_all(trackers, context)
    if not seen:
        return {}
    logger_eval.debug(f"{set}: {seen} batches ({since(started)})")
    return collect_results(trackers)
