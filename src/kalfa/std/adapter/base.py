import functools

from cirak import Deferred


def wires(keys):
    keys = keys or {}
    return keys.get("output"), keys.get("target")


def observed(value):
    detach = getattr(value, "detach", None)
    return detach() if callable(detach) else value


def rebound(function, name, value):
    keywords = dict(getattr(function, "keywords", None) or {})
    base = getattr(function, "func", function)
    keywords[name] = value
    return functools.partial(base, **keywords)


def built(function, **available):
    keywords = dict(getattr(function, "keywords", None) or {})
    if not any(isinstance(value, Deferred) for value in keywords.values()):
        return function
    base = getattr(function, "func", function)
    resolved = {key: value.build(**available) if isinstance(value, Deferred) else value
                for key, value in keywords.items()}
    return functools.partial(base, **resolved)
