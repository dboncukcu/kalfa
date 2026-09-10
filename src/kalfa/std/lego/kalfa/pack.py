from kalfa.registration import lego


@lego("/lego/kalfa/pack", aliases="items", description="A mapping of the given items")
def pack(items):
    return dict(items or {})
