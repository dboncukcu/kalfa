from cirak import register

from .kinds import cirak_kind, kind_of


def lego(uri, target=None, *, description=None, **facts):
    if "kind" in facts:
        raise ValueError(f"{uri}: kalfa.lego takes the kind from the first segment of the URI; drop kind=")
    kind = kind_of(uri)
    return register(uri, target, description=description, kind=cirak_kind(kind), **facts)


def pack(package):
    kind, provider = package.rsplit(".", 2)[-2:]
    prefix = f"/{kind}/{provider}/"

    def declare(uri, target, **facts):
        name = uri[len(prefix):] if uri.startswith(prefix) else ""
        if not name or "/" in name:
            raise ValueError(f"{uri}: {package} declares {prefix}<name>, nothing else")
        module, separator, attribute = target.partition(":")
        if not module or not separator or not attribute or "." in module:
            raise ValueError(f"{uri}: the target is 'module:name' inside {package}, got {target!r}")
        return lego(uri, f"{package}.{module}:{attribute}", **facts)

    return declare
