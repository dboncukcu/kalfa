from cirak.registry import registry


def cirak_kind(kind):
    return "predicate" if kind == "trigger" else kind


def kinds():
    declared = [kind for kind in registry.kinds if kind not in ("builder", "predicate", "data")]
    return [*declared, "builder", "data"]


def kind_of(uri):
    if not isinstance(uri, str) or not uri.startswith("/") or uri.count("/") < 2:
        raise ValueError(f"{uri!r} is not a lego URI (/<kind>/<pack>/<name>)")
    kind = uri.split("/", 2)[1]
    if kind not in kinds():
        raise ValueError(f"{uri}: {kind!r} is not a kalfa kind; the kinds are {kinds()}")
    return kind


def kalfa_kind(uri):
    try:
        return kind_of(uri)
    except ValueError:
        return None


def names_of(value):
    if value is None:
        return ()
    return (value,) if isinstance(value, str) else tuple(value)
