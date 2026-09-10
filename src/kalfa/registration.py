from cirak import register

from .kinds import cirak_kind, kind_of


def lego(uri, target=None, *, description=None, **facts):
    if "kind" in facts:
        raise ValueError(f"{uri}: kalfa.lego takes the kind from the first segment of the URI; drop kind=")
    kind = kind_of(uri)
    return register(uri, target, description=description, kind=cirak_kind(kind), **facts)
