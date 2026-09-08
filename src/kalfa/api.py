"""kalfa's commands as functions: check, run, resume, predict; collect lives in collect.py."""

import json
import random
import shutil
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy
import tezgah
import torch
from cirak.build import build_components
from cirak.errors import CirakError, CirakWarning, ConfigError, render_problems
from cirak.registry import registry

from .aliasing import aliasing_problems
from .check import Checker
from .kinds import kalfa_kind
from .config import load_surface
from .driver import recipe
from .record import (read_resolved, record_dir, resume_source, write_flow, write_resolved, write_resume_note)
from .recipe import RUN_INPUTS, analyze, compile, dump, implicit_bindings
from .std import checkpoint as checkpoints
from .std.log import Progress, sink
from .std.pre import read_prep
from .std.runtime import resolve_model


class KalfaError(CirakError):
    pass


@dataclass
class Prepared:
    surface: object
    problems: list
    sizes: dict | None = None
    header: dict | None = None
    document: dict | None = None
    analysis: object = None
    pipeline: object = None
    implicit: list = field(default_factory=list)
    loaded: dict | None = None
    aliasing: list = field(default_factory=list)

    @property
    def errors(self):
        return [problem for problem in self.problems if problem.severity == "error"]

    @property
    def warnings(self):
        return [problem for problem in self.problems if problem.severity == "warning"]

    def dump(self):
        if self.analysis is None:
            return None
        plan = self.pipeline.resolved if self.pipeline is not None else None
        return dump(self.analysis, plan)


def has_errors(problems):
    return any(problem.severity == "error" for problem in problems)


def prepare(paths, sets=None, inputs=RUN_INPUTS, dry=True) -> Prepared:
    """Load, check and compile a config: the surface, the recipe on cirak, the pipeline on tezgah."""
    surface = load_surface(paths, sets)
    checker = Checker(surface, registry)
    problems = list(surface.problems)
    blocking = has_errors(problems)
    if not blocking:
        checker.run()
        problems.extend(checker.problems)
    prepared = Prepared(surface, problems)
    prepared.header = checker.header
    prepared.sizes = checker.sizes() if checker.header is not None else None
    if blocking or has_errors(checker.problems) and not checker.header_only_errors():
        return prepared
    try:
        prepared.document = recipe(surface.data, registry, surface.aliases)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        from cirak.errors import error

        problems.append(error("driver_failed", f"the driver cannot shape this config: {exc!r}"))
        return prepared
    analysis = analyze(prepared.document)
    prepared.analysis = analysis
    problems.extend(analysis.problems)
    if has_errors(analysis.problems):
        return prepared
    pipeline, tezgah_problems = compile(analysis, inputs, dry)
    problems.extend(tezgah_problems)
    prepared.pipeline = pipeline
    if pipeline is not None and pipeline.resolved is not None:
        prepared.implicit = list(implicit_bindings(pipeline.resolved))
        prepared.aliasing = aliasing_problems(pipeline, registry)
        problems.extend(prepared.aliasing)
    return prepared


def gate(problems):
    failures = [problem for problem in problems if problem.severity == "error"]
    warns = [problem for problem in problems if problem.severity == "warning"]
    if warns:
        warnings.warn(CirakWarning(render_problems(warns)), stacklevel=2)
    if failures:
        raise ConfigError(failures)


def check(paths, sets=None, load=False) -> Prepared:
    """Check a config; with ``load`` the data block runs and the real set sizes (after filters) are reported."""
    prepared = prepare(paths, sets, dry=True)
    if load and prepared.document is not None and not prepared.errors:
        prepared.loaded = loaded_sizes(prepared.document)
    return prepared


def loaded_sizes(document):
    """The set sizes the data block of a recipe produces: the loaders are built and their datasets measured."""
    from .std.feed import dataset_size

    outputs = _data_outputs(document, ["train_loader", "valid_loader", "test_loader"])
    return {name: dataset_size(outputs[f"{name}_loader"].dataset) for name in ("train", "valid", "test")}


@dataclass
class Probe:
    sizes: dict | None = None
    prep: object = None
    features: int | None = None
    parameters: dict = field(default_factory=dict)
    shapes: dict = field(default_factory=dict)
    notes: dict = field(default_factory=dict)


def probe(document) -> Probe:
    """Run the data and models blocks of a recipe: the real set sizes, the fitted plan and the models built on one
    batch, so that lazy layers have their shapes and the parameters can be counted. Nothing is written."""
    from .std.feed import dataset_size
    from .std.runtime import call_model, named_outputs

    outputs = _flow_outputs(document, ("data", "models"),
                            ["prep", "train_loader", "valid_loader", "test_loader", "models", "composites"])
    found = Probe(sizes={name: dataset_size(outputs[f"{name}_loader"].dataset) for name in ("train", "valid", "test")},
                  prep=outputs.get("prep"))
    if found.prep is not None:
        found.features = len(found.prep.features)
    batch = next(iter(outputs["train_loader"]), None)
    models = {**(outputs.get("models") or {}), **(outputs.get("composites") or {})}
    for name, module in models.items():
        if batch is not None:
            try:
                with torch.no_grad():
                    result = named_outputs(module, call_model(module, batch))
                found.shapes[name] = {wire: tuple(value.shape) for wire, value in result.items()
                                      if hasattr(value, "shape")}
            except Exception as exc:
                found.notes[name] = f"not built on the batch: {type(exc).__name__} {exc}"
        if getattr(module, "initialized", True):
            found.parameters[name] = (sum(item.numel() for item in module.parameters()),
                                      sum(item.numel() for item in module.parameters() if item.requires_grad))
    return found


def seed_all(seed):
    if seed is None:
        return
    random.seed(seed)
    numpy.random.seed(int(seed) % (2 ** 32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


@dataclass
class RunResult:
    record: str
    report: object
    device: object = None


DEFAULT_DEVICE = "/device/kalfa/cpu"


def build_device(call):
    """A torch device from a device value: a short name (``cuda``), a ``{uri, params}`` call, or a torch.device
    (kept as it is). Returns the device, the URI and the params."""
    if isinstance(call, torch.device):
        return call, str(call), {}
    if isinstance(call, str):
        uri, params = call, {}
    elif isinstance(call, dict) and isinstance(call.get("uri"), str):
        uri, params = call["uri"], dict(call.get("params") or {})
    else:
        raise KalfaError(f"device must be a device lego (auto, cpu, cuda, mps or {{uri, params}}), got {call!r}")
    if not uri.startswith("/"):
        resolved = registry.aliases().get(uri)
        if resolved is None:
            known = sorted(name for name, target in registry.aliases().items() if kalfa_kind(target) == "device")
            raise KalfaError(f"device {uri!r} is not a known device lego; the short names are {known}")
        uri = resolved
    if registry.lookup(uri) is None:
        raise KalfaError(f"device {uri} is not registered")
    if kalfa_kind(uri) != "device":
        raise KalfaError(f"{uri} is a {kalfa_kind(uri)} lego, device needs a device lego")
    try:
        device = registry.resolve(uri)(**params)
    except RuntimeError as exc:
        raise KalfaError(str(exc)) from exc
    if not isinstance(device, torch.device):
        raise KalfaError(f"device lego {uri} returned {type(device).__name__}, not a torch.device")
    return device, uri, params


def device_of(config):
    """The torch device of a config: its device lego built (cpu when nothing is written); returns the device, the
    URI and the params."""
    return build_device(config.get("device") if config.get("device") is not None else DEFAULT_DEVICE)


def to_device(modules, device):
    """Move every module of a mapping to the device; nothing happens without one."""
    if device is None:
        return modules
    for module in modules.values():
        module.to(device)
    return modules


def write_device_note(record, device, uri, params):
    (Path(record) / "device.json").write_text(json.dumps({"device": str(device), "uri": uri, "params": params},
                                                           indent=2))


def run(paths, sets=None, executor="serial", workers=None, resume=None, resume_from=None, when=None) -> RunResult:
    """Run a config: check, compile, open the record directory, run on tezgah."""
    inputs = list(RUN_INPUTS) + (["resume"] if resume is not None else [])
    prepared = prepare(paths, sets, inputs=inputs, dry=False)
    gate(prepared.problems)
    if executor != "serial" and prepared.aliasing:
        from cirak.errors import error

        gate([error(problem.kind, problem.message, hint=problem.hint) for problem in prepared.aliasing])
    config = prepared.surface.data
    record = record_dir(config, when)
    target = Path(record)
    if target.exists() and any(target.iterdir()):
        raise KalfaError(f"record directory {record} exists and is not empty; change record or remove it")
    seed_all(config.get("seed"))
    target.mkdir(parents=True, exist_ok=True)
    write_resolved(record, prepared.surface)
    copy_plugins(config.get("plugins"), target)
    write_flow(record, prepared.dump())
    if resume_from is not None:
        old = Path(resume_from) / "checkpoints"
        if old.is_dir():
            shutil.copytree(old, target / "checkpoints", dirs_exist_ok=True)
        write_resume_note(record, resume_from, resume)
    device, uri, params = device_of(config)
    write_device_note(record, device, uri, params)
    values = {"device": device, "record": record}
    if resume is not None:
        values["resume"] = str(resume)
    tezgah.subscribe(prepared.pipeline, sink)
    try:
        report = tezgah.run(prepared.pipeline, inputs=values, executor=executor, workers=workers, record_dir=record)
    finally:
        if Progress.current is not None:
            Progress.current.close()
    return RunResult(record, report, device)


def copy_plugins(names, record):
    """Copy the plugin modules the run imported into <record>/plugins, so that predict, generate and resume on the
    record directory find them without the original config next to it."""
    import sys

    for name in names or []:
        module = sys.modules.get(name)
        file = getattr(module, "__file__", None)
        if not file:
            continue
        source = Path(file)
        target = Path(record) / "plugins"
        target.mkdir(parents=True, exist_ok=True)
        if source.name == "__init__.py":
            shutil.copytree(source.parent, target / source.parent.name, dirs_exist_ok=True)
        else:
            shutil.copyfile(source, target / source.name)


def resume(run_dir, sets=None, executor="serial", workers=None, when=None) -> RunResult:
    source = resume_source(run_dir)
    if source is None:
        raise KalfaError(f"{run_dir} has neither checkpoints/last.pt nor final/state.pt; nothing to resume")
    resolved = Path(run_dir) / "resolved.yaml"
    if not resolved.exists():
        raise KalfaError(f"{run_dir} has no resolved.yaml")
    return run([str(resolved)], sets, executor, workers, resume=source, resume_from=str(run_dir), when=when)


def rebuild_models(analysis, store, prep=None):
    """The models of a run built again from its recipe: trained models first, composites from them; ``prep`` feeds
    the kind data components among the layer params."""
    models = {}
    composites = {}
    for name, node in (analysis.flow.get("models") or {}).items():
        if not (isinstance(node, dict) and "block" in node and "builder" in node):
            continue
        expansion = store.expansion(f"flow.models.{name}")
        graph = store.graph(expansion)
        builder = registry.resolve(node["builder"])
        params = store.resolve_params(node.get("params") or {})
        inputs = node.get("inputs") or {}
        output = node["outputs"][0] if isinstance(node.get("outputs"), list) else name
        if "models" in inputs:
            composites[output] = builder(graph, models=models, prep=prep, **params)
        else:
            models[output] = builder(graph, prep=prep, **params)
    return models, composites


def weights_of(run_dir, which):
    base = Path(run_dir)
    if which == "best":
        path = base / "checkpoints" / "best.pt"
        if not path.exists():
            raise KalfaError(f"{run_dir} has no checkpoints/best.pt")
        return path
    for candidate in (base / "final" / "state.pt", base / "checkpoints" / "last.pt"):
        if candidate.exists():
            return candidate
    raise KalfaError(f"{run_dir} has neither final/state.pt nor checkpoints/last.pt")


@dataclass
class Prediction:
    path: str | None
    table: object
    model: str


def predict(run_dir, model=None, which=None, data=None, sets=None, device=None) -> Prediction:
    """Predict with a recorded run: the report model on the run's test set, or any model on new data.

    ``device`` is a device lego value (a short name or ``{uri, params}``); without it the models stay on the cpu.
    """
    from .std.eval import prediction_table
    from .std.loader import torch as torch_loader
    from .std.pre import apply

    resolved = Path(run_dir) / "resolved.yaml"
    if not resolved.exists():
        raise KalfaError(f"{run_dir} has no resolved.yaml")
    surface = load_surface([str(resolved)], sets)
    gate(surface.problems)
    config = surface.data
    which = which or config["training"].get("report", "last")
    document = recipe(config, registry, surface.aliases)
    analysis = analyze(document)
    gate(analysis.problems)
    store = build_components(analysis.data, analysis.expansions, registry)
    prep = read_prep(run_dir)
    models, composites = rebuild_models(analysis, store, prep)
    payload = checkpoints.load(weights_of(run_dir, which))
    for name, module in models.items():
        if name in payload.get("models", {}):
            module.load_state_dict(payload["models"][name])
    name = model or document["flow"]["after"]["params"].get("predicts")
    target = resolve_model(name, models, composites)
    device = build_device(device)[0] if device is not None else None
    to_device(models, device)
    to_device(composites, device)
    if data is not None:
        source = config["data"]["source"]
        df = registry.resolve(source["uri"])(**{**(source.get("params") or {}), "path": data})
        frame = apply(df, prep, "test")
        feed = document["flow"]["data"]["params"]["feed"]
        dataset = registry.resolve(feed["uri"])(frame, None, **(feed.get("params") or {}))
        loader = torch_loader(dataset, "test", document["flow"]["data"]["params"]["batch"])
        tag = f"_{Path(data).stem}"
    else:
        loader = _test_loader(document)
        tag = ""
    if model is not None:
        tag += f"_{model}"
    table = prediction_table(target, loader, prep, loader.dataset, device)
    path = Path(run_dir) / f"predictions{tag}.parquet"
    table.to_parquet(path, index=False)
    return Prediction(str(path), table, name)


@dataclass
class Generated:
    path: str | None
    samples: object


def generate(run_dir, which=None, sets=None, device=None) -> Generated:
    """Run the generate lego of a recorded run with its report models; samples land under samples/.

    ``device`` is a device lego value (a short name or ``{uri, params}``); without it the models stay on the cpu.
    """
    from .std.checkpoint import load as load_payload
    from .std.eval import write_samples
    from .std.pre import read_prep
    from .std.runtime import turn_generator

    resolved = Path(run_dir) / "resolved.yaml"
    if not resolved.exists():
        raise KalfaError(f"{run_dir} has no resolved.yaml")
    surface = load_surface([str(resolved)], sets)
    gate(surface.problems)
    config = surface.data
    if config.get("generate") is None:
        raise KalfaError(f"{run_dir}: the config has no generate section")
    which = which or config["training"].get("report", "last")
    document = recipe(config, registry, surface.aliases)
    analysis = analyze(document)
    gate(analysis.problems)
    store = build_components(analysis.data, analysis.expansions, registry)
    prep = read_prep(run_dir)
    models, composites = rebuild_models(analysis, store, prep)
    payload = load_payload(weights_of(run_dir, which))
    for name, module in models.items():
        if name in payload.get("models", {}):
            module.load_state_dict(payload["models"][name])
    everything = {**composites, **models}
    from .std.model import clone

    for name, state in payload.get("emas", {}).items():
        if name in models:
            ema = clone(models[name], 1.0)
            ema.load_state_dict(state)
            everything[f"{name}.ema"] = ema
    sampler = store.resolve_params(document["flow"]["after"]["params"]["generate"])
    device = build_device(device)[0] if device is not None else None
    to_device(everything, device)
    seed_all(config.get("seed"))
    samples = sampler(models=everything, prep=prep, rng=turn_generator(device))
    target = Path(run_dir) / "samples"
    write_samples(samples, target)
    path = target / ("samples.txt" if isinstance(samples, str) else "samples.pt")
    return Generated(str(path), samples)


def _data_outputs(document, names):
    """Outputs of the data block of a recipe run on its own (no record, nothing written)."""
    return _flow_outputs(document, ("data",), names)


def _flow_outputs(document, blocks, names):
    """Outputs of the named flow blocks of a recipe run on their own (no record, nothing written)."""
    flow = {"outputs": list(names)}
    for block in blocks:
        flow[block] = document["flow"][block]
    analysis = analyze({**{key: value for key, value in document.items() if key != "flow"}, "flow": flow})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gate(analysis.problems)
        pipeline, problems = compile(analysis, [], dry=False)
        gate(problems)
        report = tezgah.run(pipeline, inputs={})
    return report.outputs


def _test_loader(document):
    """The run's own test loader, produced by the data block of its recipe."""
    return _data_outputs(document, ["test_loader"])["test_loader"]


__all__ = ["Generated", "KalfaError", "Prepared", "Prediction", "Probe", "RunResult", "check", "generate",
           "predict", "prepare", "probe", "read_resolved", "resume", "run", "seed_all"]
