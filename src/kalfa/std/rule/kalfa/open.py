import copy

from kalfa.registration import lego


@lego("/rule/kalfa/open", description="Open the rule chain of a turn")
def open_rules(rules):
    out = copy.deepcopy(rules or {})
    out["fired"] = []
    out["pending"] = {}
    out["ready"] = list(dict.fromkeys([*(out.get("sticky") or []), *(out.get("ever") or [])]))
    return out
