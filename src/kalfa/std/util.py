"""Small std legos the templates wire with: constants, packs, the metrics merge and identity."""

import copy


from ..registration import lego
from ..kinds import HISTORY_PREFIX


@lego("/lego/kalfa/const", description="A fresh copy of a constant value")
def const(value):
    return copy.deepcopy(value)


@lego("/lego/kalfa/pack", aliases="items", description="A mapping of the given items")
def pack(items):
    return dict(items or {})


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


@lego("/lego/kalfa/identity", aliases="value", description="The value itself")
def identity(value):
    return value
