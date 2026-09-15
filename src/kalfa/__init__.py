from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from cirak import register_fragment

from . import std
from .registration import lego

PACKS = Path(__file__).parent / "packs"
CONTRACT = Path(__file__).parent / "contract.yaml"


def register_packs():
    for path in sorted(PACKS.glob("*.yaml")):
        register_fragment(f"/alias/kalfa/{path.stem}", path,
                          description=f"alias pack {path.stem}: short names for the std legos")


register_packs()

try:
    __version__ = version("kalfa")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["CONTRACT", "PACKS", "__version__", "lego", "std"]
