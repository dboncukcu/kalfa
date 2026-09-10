import inspect

from cirak.registry import registry

from .config import pack_tables
from .contract import Contract
from .kinds import kalfa_kind, kinds
from .std import STD_URIS

HEADER = """# kalfa lego reference

Generated from the registry by `kalfa docs --write DOCS.md`; do not edit by hand, the test `tests/test_docs.py`
compares this file with the registry. One table per kind: the URI, the alias names, the signature with the param
defaults, the facts a lego declares and its description. The kind of a lego is the first segment of its URI; which
config section it may be written in follows from the kind (`CONFIG.md` section 5). Every URI of the catalog is
valid in a config; the alias packs at the end give the short names. The skeleton steps come after the catalog.
"""

PLUGIN_NOTE = """Legos outside kalfa's std set: what the plugin modules of this listing registered (`kalfa docs --plugin
module` or `kalfa docs --config config.yaml`). They are written in a config exactly like the std legos, the kind is the
first segment of the URI and decides which section takes them.
"""

SKELETON_NOTE = """These are the skeleton steps `src/kalfa/contract.yaml` calls: the nodes of its blocks, the builder,
the loader, fit, read_prep and figures of its wiring, and the helpers the `sizes` and `header` facts name. They
are not written in a config; the contract places them and the driver fills their params from the config
sections. The list is derived from the URIs the contract mentions, so it cannot drift. The rest of the wiring
stays in the catalog above: the adapters (`/adapter/kalfa/criterion`, `/adapter/kalfa/metric` and
`/adapter/kalfa/objective`, which wrap the losses and metrics entries of a config by kind) and the defaults that
stand in for a config value (`/split/kalfa/random`, `/device/kalfa/cpu`, `/rng/kalfa/derived`).
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

    contract = Contract.load()
    walk(contract.blocks)
    for value in contract.wiring.values():
        if isinstance(value, str) and kalfa_kind(value) in ("lego", "loader", "builder") \
                and registry.lookup(value) is not None:
            found.add(value)
    for uri in STD_URIS:
        facts = registry.facts(uri)
        for name in ("sizes", "header"):
            if isinstance(facts.get(name), str):
                found.add(facts.get(name))
    return found


def plugin_uris():
    found = []
    for uri in sorted(registry.uris()):
        entry = registry.lookup(uri)
        if uri in STD_URIS or entry is None or entry.fragment:
            continue
        found.append(uri)
    return found


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


def render(uris=None, plugins=None):
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
        "source": "data.source", "transform": "data.transform", "split": "data.split", "pre": "data.preprocessors",
        "feed": "data.feed",
        "loader": "the contract", "layer": "model nodes", "init": "model init", "criterion": "losses, metrics",
        "objective": "losses", "metric": "metrics", "adapter": "the driver", "optimizer": "optimizers",
        "schedule": "optimizer schedule", "turn": "training.turn", "trigger": "training.stop, rules when",
        "checkpoint": "training.checkpoint", "rule": "the contract", "generate": "generate",
        "plot": "plots", "strategy": "sweep.strategy", "device": "device, predict --device, generate --device",
        "rng": "rng",
        "lego": "a param value, or the contract", "builder": "the contract",
        "data": "a param value ({uri: name})",
    }
    grouped = {kind: [uri for uri in chosen if kalfa_kind(uri) == kind] for kind in kinds()}
    for kind in kinds():
        if grouped[kind]:
            lines.append(f"| `{kind}` | {places.get(kind, '')} | {len(grouped[kind])} |")
    lines.append("")
    for kind in kinds():
        entries = grouped[kind]
        if not entries:
            continue
        lines.append(f"### {kind}\n")
        table(lines, entries)
    if plugins is not None:
        lines.append("## Plugin legos\n")
        lines.append(PLUGIN_NOTE)
        by_kind = {kind: [uri for uri in plugins if kalfa_kind(uri) == kind] for kind in kinds()}
        by_kind["other"] = [uri for uri in plugins if kalfa_kind(uri) is None]
        if not plugins:
            lines.append("Nothing registered outside kalfa's std set.\n")
        for kind, entries in by_kind.items():
            if entries:
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
