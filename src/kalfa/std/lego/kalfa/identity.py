from kalfa.registration import lego


@lego("/lego/kalfa/identity", aliases="value", description="The value itself")
def identity(value):
    return value
