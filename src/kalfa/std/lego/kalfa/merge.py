from kalfa.kinds import HISTORY_PREFIX
from kalfa.registration import lego


@lego("/lego/kalfa/merge", description="Merge the per set metrics under train/, val/ and test/")
def merge(parts):
    merged = {}
    order = list(HISTORY_PREFIX)
    keys = sorted(parts or {}, key=lambda key: (order.index(key.removesuffix("_metrics"))
                                                 if key.removesuffix("_metrics") in order else len(order), key))
    for key in keys:
        set_name = key.removesuffix("_metrics")
        prefix = HISTORY_PREFIX.get(set_name, set_name)
        for name, value in (parts[key] or {}).items():
            merged[f"{prefix}/{name}"] = value
    return merged
