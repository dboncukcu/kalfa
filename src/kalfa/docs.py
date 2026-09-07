"""kalfa docs: the lego reference generated from the registry (the std legos and the alias packs)."""

import inspect
from pathlib import Path

from cirak.registry import registry

from .kinds import KINDS, kalfa_kind

HEADER = """# kalfa lego reference

Generated from the registry by `kalfa docs --write DOCS.md`; do not edit by hand, the test `tests/test_docs.py`
compares this file with the registry. One table per kind: the URI, the alias names, the signature with the param
defaults, the facts a lego declares and its description. The kind of a lego is the first segment of its URI; which
config section it may be written in follows from the kind (`CONFIG.md` section 5). Every URI of the catalog is
valid in a config; the alias packs at the end give the short names. The skeleton steps come after the catalog.
"""

SKELETON_NOTE = """These are the skeleton steps `src/kalfa/templates/kalfa.yaml` calls; they are not written in a config, the
template places them and the driver fills their params from the config sections. The list is derived from the URIs
the template mentions, so it cannot drift. Two more legos are inserted by the driver rather than by the template
and stay in the catalog above: the adapters (`/adapter/kalfa/criterion` and `/adapter/kalfa/metric`, which wrap the
criteria and metrics of a config) and the progress component (`/lego/kalfa/progress`).
"""


def signature_text(target):
    try:
        return str(inspect.signature(target))
    except (TypeError, ValueError):
        return ""


def facts_text(facts):
    parts = []
    for name, value in facts.declared().items():
        if name in ("kind", "alias"):
            continue
        if isinstance(value, list):
            parts.append(f"{name}: {', '.join(str(item) for item in value)}")
        elif isinstance(value, dict):
            parts.append(f"{name}: " + ", ".join(f"{key}={item}" for key, item in value.items()))
        else:
            parts.append(f"{name}: {value}")
    return "; ".join(parts)


def cell(text):
    return str(text).replace("|", "\\|").replace("\n", " ")


def template_uris():
    """Every registered URI the flow template mentions: the skeleton steps of a run."""
    from ruamel.yaml import YAML

    from . import TEMPLATE

    found = set()

    def walk(value):
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str) and value.startswith("/") and registry.lookup(value) is not None:
            found.add(value)

    walk(YAML(typ="safe").load(Path(TEMPLATE).read_text()))
    return found


def pack_tables():
    from ruamel.yaml import YAML

    tables = {}
    for uri, path in sorted(registry.fragments().items()):
        if uri.startswith("/alias/"):
            tables[uri] = (YAML(typ="safe").load(Path(path).read_text()) or {}).get("alias") or {}
    return tables


def table(lines, uris):
    lines.append("| URI | Alias | Signature | Facts | Description |")
    lines.append("|---|---|---|---|---|")
    for uri in uris:
        entry = registry.lookup(uri)
        aliases = ", ".join(f"`{name}`" for name in entry.facts.alias)
        signature = signature_text(entry.target)
        lines.append(f"| `{uri}` | {aliases} | `{cell(signature)}` | {cell(facts_text(entry.facts))} | "
                     f"{cell(entry.description or '')} |")
    lines.append("")


def render(uris=None):
    """The reference as Markdown: the catalog by kind, the skeleton steps the template calls, the alias packs."""
    from .std import STD_URIS

    everything = sorted(uris if uris is not None else STD_URIS)
    skeleton = sorted(uri for uri in everything if uri in template_uris())
    chosen = [uri for uri in everything if uri not in set(skeleton)]
    packs = pack_tables()
    lines = [HEADER]
    lines.append("## Catalog\n")
    lines.append("The legos a config writes, by kind.\n")
    lines.append("### Kinds\n")
    lines.append("| Kind | Where it is written | Count |")
    lines.append("|---|---|---|")
    places = {
        "source": "data.source", "split": "data.split", "pre": "data.preprocessors", "feed": "data.feed",
        "loader": "the template", "layer": "model nodes", "init": "model init", "criterion": "losses, metrics",
        "objective": "losses", "metric": "metrics", "adapter": "the driver", "optimizer": "optimizers",
        "schedule": "optimizer schedule", "turn": "training.turn", "trigger": "training.stop, rules when",
        "checkpoint": "training.checkpoint", "rule": "the template", "generate": "generate",
        "plot": "plots", "strategy": "sweep.strategy", "device": "device, predict --device, generate --device",
        "lego": "a param value, or the driver", "builder": "the template",
        "data": "a param value ({uri: name})",
    }
    grouped = {kind: [uri for uri in chosen if kalfa_kind(uri) == kind] for kind in KINDS}
    for kind in KINDS:
        if grouped[kind]:
            lines.append(f"| `{kind}` | {places.get(kind, '')} | {len(grouped[kind])} |")
    lines.append("")
    for kind in KINDS:
        entries = grouped[kind]
        if not entries:
            continue
        lines.append(f"### {kind}\n")
        table(lines, entries)
    lines.append("## Skeleton steps\n")
    lines.append(SKELETON_NOTE)
    table(lines, skeleton)
    lines.append("## Alias packs\n")
    for uri, members in packs.items():
        lines.append(f"### {uri}\n")
        lines.append("| Alias | URI | Kind |")
        lines.append("|---|---|---|")
        for name, target in members.items():
            lines.append(f"| `{name}` | `{target}` | {kalfa_kind(target) or ''} |")
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"
