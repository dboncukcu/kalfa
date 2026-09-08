"""kalfa's standard legos, registered with cirak on import; the kinds and the facts are declared first."""

from cirak import declare_facts, declare_kinds
from cirak.registry import registry

from ..kinds import FACTS, KINDS

declare_kinds(*[kind for kind in KINDS if kind not in ("builder", "data")])
declare_facts(*FACTS)

_before = set(registry.uris())

from . import (adapter, builder, checkpoint, criterion, data, device, eval, feed, generate, init, layer, loader,  # noqa: E402
               log, metric, model, objective, optimizer, plot, pre, rule, schedule, source, split, strategy, trigger,
               turn, util)

STD_URIS = frozenset(set(registry.uris()) - _before)

__all__ = ["STD_URIS", "adapter", "builder", "checkpoint", "criterion", "data", "device", "eval", "feed", "generate", "init", "layer",
           "loader", "log", "metric", "model", "objective", "optimizer", "plot", "pre", "rule", "schedule", "source", "split", "strategy",
           "trigger", "turn", "util"]
