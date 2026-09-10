import copy

from kalfa.registration import lego


@lego("/lego/kalfa/const", description="A fresh copy of a constant value")
def const(value):
    return copy.deepcopy(value)
