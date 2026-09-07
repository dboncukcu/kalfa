"""The bridge to cirak: the driver document opened over the template as a layer, compiled and dumped."""

from cirak.api import Analysis, compile_checked, dump_document, dump_text
from cirak.expand import expand, expand_flow
from cirak.loader import Layer, LoadedFile, load
from cirak.merge import merge_layers
from cirak.registry import registry
from cirak.resolve import resolve as resolve_values
from cirak.validate import validate

from . import TEMPLATE

RUN_INPUTS = ("device", "record")
DRIVER_LABEL = "kalfa driver"


def analyze(document) -> Analysis:
    """The cirak analysis of a driver document layered above the template, the way cirak.api.analyze works on files."""
    template, problems = load([str(TEMPLATE)], registry.fragments())
    layer = Layer(files=[LoadedFile(DRIVER_LABEL, document, {})], below=[template], label=DRIVER_LABEL)
    data, provenance, overrides, merge_problems = merge_layers(layer)
    problems = [*problems, *merge_problems]
    data, resolve_problems = resolve_values(data, provenance)
    flow, flow_provenance, flow_problems = expand_flow(data, provenance)
    analysis = Analysis(data, provenance, {}, [], layer, overrides, flow, flow_provenance)
    expansions, expand_problems = expand(data, provenance, flow, analysis.view_provenance)
    collected = [*problems, *resolve_problems, *flow_problems, *expand_problems]
    collected += validate(analysis.view, analysis.view_provenance, expansions, registry)
    return Analysis(data, provenance, expansions, collected, layer, overrides, flow, flow_provenance)


def compile(analysis, inputs, dry):
    """The tezgah pipeline of an analysis and the problems tezgah found; ``inputs`` names the root bus keys."""
    return compile_checked(analysis, list(inputs), dry=dry)


def dump(analysis, plan=None) -> str:
    return dump_text(dump_document(analysis, plan))


def recipe_text(document) -> str:
    """A driver document as YAML, lego records inline the way the dumps write them."""
    from io import StringIO

    from cirak.api import _plain_tree
    from ruamel.yaml import YAML

    yaml = YAML()
    yaml.width = 120
    stream = StringIO()
    yaml.dump(_plain_tree(document), stream)
    return stream.getvalue()


def implicit_bindings(plan, path=""):
    """Every implicit binding of the resolved pipeline as (node path, parameter, key) triples."""
    table = getattr(plan, "table", None) or {}
    for name, entry in table.items():
        child = f"{path}.{name}" if path else name
        kind = getattr(entry, "kind", None)
        if kind == "step":
            for param, key in (entry.implicit or {}).items():
                yield child, param, key if isinstance(key, str) else ", ".join(key)
            for param, key in (getattr(entry, "when_implicit", None) or {}).items():
                yield f"{child} when", param, key
        elif kind in ("loop", "map"):
            for param, key in (getattr(entry, "until_implicit", None) or {}).items():
                yield f"{child} until", param, key
            yield from implicit_bindings(entry.body, f"{child}.body")
        elif kind == "pipeline":
            yield from implicit_bindings(entry, child)
        elif kind == "branch":
            for label, case in (entry.cases or {}).items():
                yield from implicit_bindings(case, f"{child}.{label}")
