from kalfa.registration import lego


@lego("/rule/kalfa/effects", returns="effects",
      description="The effects the fired rules left for this turn")
def effects(rules):
    return dict((rules or {}).get("effects") or {})
