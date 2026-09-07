"""kalfa: YAML front end for PyTorch training on top of cirak and tezgah."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from cirak import register_fragment

from . import std  # noqa: F401  registers the std legos and declares the kinds
from .registration import lego

PACKS = Path(__file__).parent / "packs"
TEMPLATE = Path(__file__).parent / "templates" / "kalfa.yaml"


def _register_packs():
    for path in sorted(PACKS.glob("*.yaml")):
        register_fragment(f"/alias/kalfa/{path.stem}", path,
                          description=f"alias pack {path.stem}: short names for the std legos")


_register_packs()

try:
    __version__ = version("kalfa")
except PackageNotFoundError:
    __version__ = "0.2.0"

__all__ = ["PACKS", "TEMPLATE", "__version__", "lego", "std"]
