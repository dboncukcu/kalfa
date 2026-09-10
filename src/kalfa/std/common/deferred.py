import inspect

from torch import nn
from cirak import Deferred


def is_deferred(value):
    return isinstance(value, Deferred)


class DeferredLayer(nn.Module):
    def __init__(self, factory, params):
        super().__init__()
        self.factory = factory
        self.params = params

    def build(self, **available):
        resolved = {name: build_deferred(value, **available) if is_deferred(value) else value
                    for name, value in self.params.items()}
        return self.factory(**resolved)


def later(factory, **params):
    if any(is_deferred(value) for value in params.values()):
        return DeferredLayer(factory, params)
    return factory(**params)


def build_deferred(deferred, **available):
    try:
        names = set(inspect.signature(deferred.target).parameters)
    except (TypeError, ValueError):
        names = set(available)
    given = {name: value for name, value in available.items() if name in names}
    missing = [name for name in names if name not in given and name not in deferred.params
               and inspect.signature(deferred.target).parameters[name].default is inspect.Parameter.empty]
    if missing:
        raise ValueError(f"{deferred.uri} needs {missing}, which the model builder cannot supply (it has "
                         f"{sorted(available)})")
    return deferred.build(**given)
