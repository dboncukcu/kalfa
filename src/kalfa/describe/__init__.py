from .text import Plain, visible


DEFAULT_SECTIONS = ("summary", "data", "model", "training", "after", "columns")
ALL_SECTIONS = (*DEFAULT_SECTIONS, "wiring")


__all__ = ["ALL_SECTIONS", "DEFAULT_SECTIONS", "Plain", "visible"]
