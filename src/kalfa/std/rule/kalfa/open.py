import copy

from kalfa.registration import lego


@lego("/rule/kalfa/open", description="Open the rule chain of a turn")
def open(rules):
    out = copy.deepcopy(rules or {})
    out["fired"] = []
    out["pending"] = {}
    out["ready"] = list(out.get("sticky") or [])
    return out
