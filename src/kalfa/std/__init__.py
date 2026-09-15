from importlib import import_module
from pathlib import Path

from cirak import declare_facts, declare_kinds
from cirak.registry import registry


declare_kinds("source", "transform", "split", "frame", "pre", "feed", "loader", "layer", "init", "criterion",
              "objective", "metric", "adapter", "optimizer", "schedule", "turn", "trigger", "checkpoint", "rule",
              "generate", "plot", "strategy", "device", "rng", "export", "calibrate", "lego")
declare_facts("uses", "needs_grad", "needs_models", "extras", "grouped", "requires", "sizes", "header", "stream",
              "samples", "needs_table", "counts", "writes", "enumerates", "describe", "roles", "needs")


def discover():
    before = set(registry.uris())
    root = Path(__file__).parent
    for path in sorted(root.glob("*/*/__init__.py")):
        import_module(f"{__name__}.{path.parent.parent.name}.{path.parent.name}")
    return frozenset(set(registry.uris()) - before)


STD_URIS = discover()
