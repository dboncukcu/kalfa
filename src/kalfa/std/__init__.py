from importlib import import_module
from pathlib import Path

from cirak import declare_facts, declare_kinds
from cirak.registry import registry

from kalfa.kinds import FACTS, KINDS

declare_kinds(*[kind for kind in KINDS if kind not in ("builder", "data")])
declare_facts(*FACTS)


def discover():
    before = set(registry.uris())
    root = Path(__file__).parent
    for path in sorted(root.glob("*/*/*.py")):
        if path.name not in ("__init__.py", "base.py"):
            import_module(f"{__name__}.{path.parent.parent.name}.{path.parent.name}.{path.stem}")
    return frozenset(set(registry.uris()) - before)


STD_URIS = discover()
