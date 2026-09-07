"""The kalfa registration decorator: cirak's ``register`` with the kind taken from the URI."""

from cirak import register

from .kinds import cirak_kind, kind_of


def lego(uri, target=None, *, description=None, **facts):
    """Register a lego under ``uri``; its kind is the first segment of the URI (``/criterion/...``, ``/turn/...``,
    ``/data/...``) and must be one of ``kalfa.kinds.KINDS``. ``/trigger/`` legos are cirak predicates; ``/builder/``
    and ``/data/`` keep cirak's own kind names."""
    if "kind" in facts:
        raise ValueError(f"{uri}: kalfa.lego takes the kind from the first segment of the URI; drop kind=")
    kind = kind_of(uri)
    return register(uri, target, description=description, kind=cirak_kind(kind), **facts)
