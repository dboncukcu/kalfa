"""Layer params that are kind data components: the layer is built once the builder has the data they need."""

import inspect

from torch import nn


def is_deferred(value):
    from cirak import Deferred

    return isinstance(value, Deferred)


class DeferredLayer(nn.Module):
    """A layer whose params include a kind data component; the model builder builds it with prep and the train
    loader at hand."""

    def __init__(self, factory, params):
        super().__init__()
        self.factory = factory
        self.params = params

    def build(self, **available):
        resolved = {name: build_deferred(value, **available) if is_deferred(value) else value
                    for name, value in self.params.items()}
        return self.factory(**resolved)


def later(factory, **params):
    """``factory(**params)`` now, or a DeferredLayer when a param is a kind data component."""
    if any(is_deferred(value) for value in params.values()):
        return DeferredLayer(factory, params)
    return factory(**params)


def build_deferred(deferred, **available):
    """Build a kind data component with the data its signature asks for (prep, loader, train_loader)."""
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
