import hashlib
import json
import shutil
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pandas
import tezgah
import torch
from cirak.api import Analysis
from cirak.build import ComponentStore, build_components
from cirak.errors import CirakWarning, ConfigError, error, render_problems
from cirak.registry import registry

from .aliasing import aliasing_problems
from .check import Checker
from .config import Surface, load_surface
from .contract import Contract
from .driver import call_with_params, recipe
from .errors import KalfaError
from .kinds import kalfa_kind
from .record import Record, read_resolved, record_dir, resume_source, write_flow, write_resolved, write_resume_note
from .recipe import analyze, compile, dump, implicit_bindings
from .std.calibrate.base import read_calibrations
from .std.checkpoint.base import load, load_into
from .std.common.device import Device
from .std.common.generation import write_samples
from .std.common.history import History
from .std.common.log import Monitor, clock, logger_for, since
from .std.common.prediction import prediction_table
from .std.common.files import atomic
from .std.common.rng import seed_all
from .std.export.base import traced_inputs
from .std.common.runtime import call_model, named_outputs, resolve_model
from .std.frame.base import read_frames, write_frames
from .std.lego.kalfa.apply import apply
from .std.lego.kalfa.apply_frames import apply_frames
from .std.lego.kalfa.clone import Ema
from .std.lego.kalfa.run_all import run_all
from .std.lego.kalfa.select import select
from .std.pre.base import Prep, read_prep, write_prep


logger = logger_for("run")


@dataclass
class Prepared:
    surface: Surface
    problems: list
    contract: Contract
    sizes: dict | None = None
    header: dict | None = None
    document: dict | None = None
    analysis: Analysis | None = None
    pipeline: tezgah.Pipeline | None = None
    implicit: list = field(default_factory=list)
    loaded: dict | None = None
    aliasing: list = field(default_factory=list)
    sets: list = field(default_factory=list)

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


def resolved_of(run_dir):
    resolved = Path(run_dir) / "resolved.yaml"
    if not resolved.exists():
        raise KalfaError(f"{run_dir} has no resolved.yaml")
    return resolved


def record_paths(paths, contract=None):
    found = []
    for path in paths:
        if not isinstance(path, dict) and Path(path).is_dir():
            found.append(str(resolved_of(path)))
            contract = recorded_contract(path, contract)
        else:
            found.append(path)
    return found, contract


def data_hash(config):
    return hashlib.sha256(json.dumps(config.get("data"), sort_keys=True, default=str).encode("utf-8")).hexdigest()


def prepared_manifest(directory):
    manifest = Record(directory).read_json("manifest.json")
    if manifest is None or manifest.get("kind") != "data":
        raise KalfaError(f"{directory} is no prepared directory; kalfa prepare cfg.yaml --out {directory} writes one")
    return manifest


def prepare(paths, sets=None, inputs=None, dry=True, contract=None, prepared=None) -> Prepared:
    started = clock()
    paths, contract = record_paths(paths, contract)
    contract = contract or Contract.load()
    inputs = contract.run_inputs if inputs is None else list(inputs)
    surface = load_surface(paths, sets, contract)
    manifest = prepared_manifest(prepared) if prepared is not None else None
    checker = Checker(surface, registry, contract, manifest["header"] if manifest is not None else None)
    problems = list(surface.problems)
    blocking = has_errors(problems)
    if not blocking:
        checker.run()
        problems.extend(checker.problems)
        logger.debug(f"checked in {since(started)}: {len(problems)} problems")
    found = Prepared(surface, problems, contract)
    found.sets = list(checker.sets)
    found.header = checker.header
    found.sizes = checker.sizes() if checker.header is not None else None
    if blocking or has_errors(checker.problems) and not checker.header_only_errors():
        return found
    if manifest is not None and manifest.get("hash") != data_hash(surface.data):
        problems.append(error("prepared_mismatch", f"{prepared} was prepared from another data section; prepare "
                                                   f"it again from this config"))
        return found
    try:
        found.document = recipe(surface.data, registry, surface.aliases, contract, prepared=prepared)
    except (KeyError, TypeError, ValueError, AttributeError) as exception:
        problems.append(error("driver_failed", f"the driver cannot shape this config: {exception!r}"))
        return found
    analysis = analyze(found.document, contract)
    found.analysis = analysis
    problems.extend(analysis.problems)
    if has_errors(analysis.problems):
        return found
    built = clock()
    pipeline, tezgah_problems = compile(analysis, inputs, dry)
    problems.extend(tezgah_problems)
    logger.debug(f"compiled the flow in {since(built)}")
    found.pipeline = pipeline
    if pipeline is not None and pipeline.resolved is not None:
        found.implicit = list(implicit_bindings(pipeline.resolved))
        found.aliasing = aliasing_problems(pipeline, registry)
        problems.extend(found.aliasing)
    return found


def gate(problems):
    failures = [problem for problem in problems if problem.severity == "error"]
    warns = [problem for problem in problems if problem.severity == "warning"]
    if warns:
        warnings.warn(CirakWarning(render_problems(warns)), stacklevel=2)
    if failures:
        raise ConfigError(failures)


def check(paths, sets=None, load=False, contract=None, prepared=None) -> Prepared:
    found = prepare(paths, sets, dry=True, contract=contract, prepared=prepared)
    if load and found.document is not None and not found.errors:
        found.loaded = (dict(prepared_manifest(prepared)["sizes"]) if prepared is not None
                        else loaded_sizes(found.document, found.contract))
    return found


@dataclass
class PreparedData:
    directory: str
    sizes: dict
    manifest: dict


def prepare_data(paths, sets=None, out=None, contract=None) -> PreparedData:
    found = prepare(paths, sets, dry=True, contract=contract)
    gate(found.problems)
    config = found.surface.data
    source = config["data"]["source"]
    if registry.facts(source["uri"]).get("stream"):
        raise KalfaError("a stream source is lazy already; there is nothing to prepare")
    names = document_sets(found.document)
    wanted = ["prep", "frames", "data_report", *[f"{name}_df" for name in names]]
    outputs = flow_outputs(found.document, ("data",), wanted, found.contract)
    target = Record(out)
    target.directory.mkdir(parents=True, exist_ok=True)
    samples = bool(registry.facts(source["uri"]).get("samples"))
    sizes = {}
    for name in names:
        part = outputs[f"{name}_df"]
        sizes[name] = int(len(part))
        if samples:
            target.write_text(f"{name}.json", json.dumps([int(position) for position in part.positions]))
        else:
            with atomic(target.path(f"{name}.parquet")) as temporary:
                part.rename_axis("row").reset_index().to_parquet(temporary, index=False)
    write_prep(outputs["prep"], target.directory)
    write_frames(outputs["frames"], target.directory)
    target.write_json("data.json", outputs["data_report"])
    manifest = target.manifest("data", hash=data_hash(config), sets=names, sizes=sizes,
                               layout="samples" if samples else "table", header=found.header,
                               source=call_with_params(source), config=[str(path) for path in paths
                                                                         if not isinstance(path, dict)])
    logger.info(f"prepared {out}: " + ", ".join(f"{name} {size}" for name, size in sizes.items()))
    return PreparedData(str(out), sizes, manifest)


def document_sets(document):
    return list(document["flow"]["data"]["params"]["sets"])


def loaded_sizes(document, contract=None):
    sets = document_sets(document)
    outputs = data_outputs(document, [f"{name}_loader" for name in sets], contract)
    return {name: outputs[f"{name}_loader"].dataset.count() for name in sets}


@dataclass
class Probe:
    sizes: dict | None = None
    prep: Prep | None = None
    features: int | None = None
    parameters: dict = field(default_factory=dict)
    shapes: dict = field(default_factory=dict)
    notes: dict = field(default_factory=dict)


def probe(document, contract=None) -> Probe:
    sets = document_sets(document)
    outputs = flow_outputs(document, ("data", "models"),
                           ["prep", *[f"{name}_loader" for name in sets], "models", "composites"], contract)
    found = Probe(sizes={name: outputs[f"{name}_loader"].dataset.count() for name in sets},
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
                                      if isinstance(value, torch.Tensor)}
            except Exception as exception:
                found.notes[name] = f"not built on the batch: {type(exception).__name__} {exception}"
        if module.initialized:
            found.parameters[name] = (sum(item.numel() for item in module.parameters()),
                                      sum(item.numel() for item in module.parameters() if item.requires_grad))
    return found


@dataclass
class RunResult:
    record: str
    report: tezgah.Report
    device: Device | None = None


def build_device(call):
    if isinstance(call, Device):
        return call
    if isinstance(call, torch.device):
        return Device(call, str(call))
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
    except RuntimeError as exception:
        raise KalfaError(str(exception)) from exception
    if not isinstance(device, torch.device):
        raise KalfaError(f"device lego {uri} returned {type(device).__name__}, not a torch.device")
    return Device(device, uri, params)


def device_of(config, contract=None):
    contract = contract or Contract.load()
    return build_device(config.get("device") if config.get("device") is not None
                        else contract.wiring["default_device"])


def write_device_note(record, device):
    (Path(record) / "device.json").write_text(json.dumps(device.note(), indent=2))


def git_note(root):
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True,
                                check=True)
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit.stdout.strip(), "dirty": bool(status.stdout.strip())}


def write_git_note(record, paths):
    files = [path for path in paths if not isinstance(path, dict)]
    root = Path(files[0]).resolve().parent if files else Path.cwd()
    (Path(record) / "git.json").write_text(json.dumps(git_note(root), indent=2))


def run(paths, sets=None, executor="serial", workers=None, resume=None, resume_from=None, when=None,
        contract=None, monitor=None, prepared=None, identity=None) -> RunResult:
    started = clock()
    paths, contract = record_paths(paths, contract)
    contract = contract or Contract.load()
    monitor = monitor or Monitor()
    inputs = contract.run_inputs + (["resume"] if resume is not None else [])
    found = prepare(paths, sets, inputs=inputs, dry=False, contract=contract, prepared=prepared)
    gate(found.problems)
    if executor != "serial" and found.aliasing:
        gate([error(problem.kind, problem.message, hint=problem.hint) for problem in found.aliasing])
    config = found.surface.data
    record = record_dir(config, when)
    target = Path(record)
    if target.exists() and any(target.iterdir()):
        raise KalfaError(f"record directory {record} exists and is not empty; change record or remove it")
    seed_all(config.get("seed"))
    if config.get("seed") is not None:
        logger.debug(f"seed {config['seed']}")
    logger.info(f"record {record}")
    target.mkdir(parents=True, exist_ok=True)
    note = Record(record)
    note.manifest(**{"kind": "run", "name": target.name, "config": [str(path) for path in paths
                                                                    if not isinstance(path, dict)],
                     "params": dict(config.get("params") or {}), "contract": contract.digest(),
                     "prepared": str(prepared) if prepared is not None else None, **(identity or {})})
    note.host()
    write_resolved(record, found.surface)
    contract.write(target / "contract.yaml")
    copy_plugins(config.get("plugins"), target)
    write_flow(record, found.dump())
    if prepared is not None:
        shutil.copytree(Path(prepared) / "fitted", target / "fitted", dirs_exist_ok=True)
    if resume_from is not None:
        old = Path(resume_from) / "checkpoints"
        if old.is_dir():
            shutil.copytree(old, target / "checkpoints", dirs_exist_ok=True)
        write_resume_note(record, resume_from, resume)
        logger.info(f"resuming {resume_from} from {Path(resume).name}")
    device = device_of(config, contract)
    logger.info(f"device {device} ({device.uri})")
    write_device_note(record, device)
    write_git_note(record, paths)
    monitor.open(record)
    values = {"device": device, "record": record, "monitor": monitor}
    if resume is not None:
        values["resume"] = str(resume)
    tezgah.subscribe(found.pipeline, monitor.sink)
    try:
        report = tezgah.run(found.pipeline, inputs=values, executor=executor, workers=workers, record_dir=record)
        history = report.outputs.get("history") or []
        monitor.summary(config.get("params"), history[-1] if history else None)
    finally:
        monitor.finish()
    logger.info(f"finished in {since(started)}")
    return RunResult(record, report, device)


def copy_plugins(names, record):
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


def recorded_contract(run_dir, contract=None):
    if contract is not None:
        return contract
    copy = Path(run_dir) / "contract.yaml"
    return Contract.load(copy) if copy.exists() else Contract.load()


def resume(run_dir, sets=None, executor="serial", workers=None, when=None, contract=None,
           monitor=None) -> RunResult:
    source = resume_source(run_dir)
    if source is None:
        raise KalfaError(f"{run_dir} has neither checkpoints/last.pt nor final/state.pt; nothing to resume")
    resolved = resolved_of(run_dir)
    return run([str(resolved)], sets, executor, workers, resume=source, resume_from=str(run_dir), when=when,
               contract=recorded_contract(run_dir, contract), monitor=monitor)


def rebuild_models(analysis, store, prep=None):
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
    if which == "final":
        path = base / "final" / "state.pt"
        if not path.exists():
            raise KalfaError(f"{run_dir} has no final/state.pt; the run has not ended")
        return path
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
class Opened:
    run_dir: str
    contract: Contract
    surface: Surface
    which: str
    document: dict
    analysis: Analysis
    store: ComponentStore
    prep: Prep
    frames: list = field(default_factory=list)
    calibrations: dict = field(default_factory=dict)
    models: dict | None = None
    composites: dict | None = None
    payload: dict | None = None

    @property
    def config(self):
        return self.surface.data

    @property
    def after(self):
        return self.document["flow"]["after"]["params"] or {}

    def rebuild(self):
        if self.models is None:
            self.models, self.composites = rebuild_models(self.analysis, self.store, self.prep)
            self.payload = load(weights_of(self.run_dir, self.which))
            for name, module in self.models.items():
                if name in self.payload.get("models", {}):
                    module.load_state_dict(self.payload["models"][name])
        return self.models, self.composites


def open_record(run_dir, which=None, sets=None, contract=None) -> Opened:
    resolved = resolved_of(run_dir)
    contract = recorded_contract(run_dir, contract)
    surface = load_surface([str(resolved)], sets, contract)
    gate(surface.problems)
    config = surface.data
    document = recipe(config, registry, surface.aliases, contract, record=run_dir)
    analysis = analyze(document, contract)
    gate(analysis.problems)
    store = build_components(analysis.data, analysis.expansions, registry)
    return Opened(str(run_dir), contract, surface, which or config["training"].get("report", "last"), document,
                  analysis, store, read_prep(run_dir), read_frames(run_dir), read_calibrations(run_dir))


def record_loaders(document, contract=None):
    sets = document_sets(document)
    outputs = data_outputs(document, [f"{name}_loader" for name in sets], contract)
    return {name: outputs[f"{name}_loader"] for name in sets}


def new_loader(opened, data):
    if isinstance(data, pandas.DataFrame):
        df = data
    else:
        source = opened.config["data"]["source"]
        df = registry.resolve(source["uri"])(**{**(source.get("params") or {}), "path": data})
    params = opened.document["flow"]["data"]["params"]
    df = apply_frames(replay_transforms(opened, df, params), opened.frames)
    frame = apply(df, opened.prep, "test", mask=params.get("mask"))
    feed = params["feed"]
    dataset = registry.resolve(feed["uri"])(frame, None, **(feed.get("params") or {}))
    spec = params["loaders"]["test"]
    return registry.resolve(spec["uri"])(dataset, **spec["params"])


def replay_transforms(opened, df, params):
    steps = [*(params.get("transform_pre") or []),
             *((params.get("set_transforms") or {}).get("test") or {}).get("transforms", [])]
    before = len(df) if isinstance(df, pandas.DataFrame) else None
    for step in steps:
        df = opened.store.resolve_params(step)(df)
    if steps and before is not None:
        logger.info(f"{len(steps)} transforms replayed, {before - len(df)} of {before} rows dropped")
    return df


def chosen_plots(table, chosen):
    if chosen == "all":
        return dict(table)
    names = [name for name in chosen.split(",") if name] if isinstance(chosen, str) else list(chosen)
    unknown = [name for name in names if name not in table]
    if unknown:
        raise KalfaError(f"plots {unknown} are not in the plots section, which has {sorted(table)}")
    return {name: table[name] for name in names}


def draw_plots(opened, chosen, predictions, history, models, bus, suffix=""):
    after = opened.after
    table = opened.store.get("plots") if opened.document.get("plots") else {}
    plots = chosen_plots(table, chosen)
    figures = opened.store.resolve_params(after["figures"])
    run_all(predictions, history, models, plots, keys=after.get("plots_keys"), predicts=after.get("predicts"),
            bus={key: bus.get(key) for key in opened.contract.plot_bus}, record=opened.run_dir, figures=figures,
            suffix=suffix)
    return list(plots)


@dataclass
class Prediction:
    path: str | None
    table: pandas.DataFrame
    model: str
    plots: list = field(default_factory=list)


def predict(run_dir, model=None, which=None, data=None, sets=None, device=None, contract=None,
            plots=None) -> Prediction:
    opened = open_record(run_dir, which, sets, contract)
    models, composites = opened.rebuild()
    name = model or opened.after.get("predicts")
    target = resolve_model(name, models, composites)
    logger.info(f"predicting with {name} ({opened.which} weights)")
    device = build_device(device) if device is not None else Device.cpu()
    device.place(models)
    device.place(composites)
    if data is not None:
        loaders = {"test": new_loader(opened, data)}
        tag = "_frame" if isinstance(data, pandas.DataFrame) else f"_{Path(data).stem}"
    else:
        loaders = record_loaders(opened.document, opened.contract)
        tag = ""
    if model is not None:
        tag += f"_{model}"
    loader = loaders["test"]
    table = prediction_table(target, loader, opened.prep, loader.dataset, device, opened.after.get("targets"))
    for item in opened.calibrations.values():
        table = item.apply(table)
    path = Path(run_dir) / f"predictions{tag}.parquet"
    table.to_parquet(path, index=False)
    logger.info(f"{len(table)} rows -> {path}")
    drawn = []
    if plots is not None:
        bus = {"prep": opened.prep, "composites": composites, "device": device,
               **{f"{set_name}_loader": item for set_name, item in loaders.items()}}
        drawn = draw_plots(opened, plots, table, History.read(run_dir), {**composites, **models}, bus, tag)
    return Prediction(str(path), table, name, drawn)


@dataclass
class Plots:
    record: str
    names: list


def plots(run_dir, only=None, sets=None, device=None, contract=None) -> Plots:
    opened = open_record(run_dir, None, sets, contract)
    contract = opened.contract
    wanted = ["prep", "models", "emas", "composites", "optimizers", "data_report", "train_df", "train_frame",
              *[f"{name}_loader" for name in document_sets(opened.document)]]
    outputs = flow_outputs(opened.document, ("data", "models", "optimizers"), wanted, contract)
    counters = {}
    rules = {}
    final = Path(run_dir) / "final" / "state.pt"
    if final.exists():
        load_into(outputs["models"], outputs["optimizers"], outputs["emas"], counters, rules, load(final))
    selected = select(outputs["models"], outputs["emas"], opened.which, str(run_dir))
    device = build_device(device) if device is not None else Device.cpu()
    device.place(selected)
    device.place(outputs["composites"])
    table = Path(run_dir) / "predictions.parquet"
    predictions = pandas.read_parquet(table) if table.exists() else None
    logger.info(f"redrawing the plots of {run_dir} with the {opened.which} models")
    bus = {**outputs, "device": device, "counters": counters, "rules": rules,
           "losses": opened.store.get("losses") if opened.document.get("losses") else {},
           "losses_keys": opened.document["flow"]["training"]["params"].get("losses_keys")}
    drawn = draw_plots(opened, only or "all", predictions, History.read(run_dir), selected, bus)
    return Plots(str(run_dir), drawn)


@dataclass
class Exported:
    path: str | None
    model: str
    format: str


def export(run_dir, format="state_dict", model=None, which=None, out=None, sets=None, device=None,
           contract=None) -> Exported:
    opened = open_record(run_dir, which, sets, contract)
    uri = format if format.startswith("/") else registry.aliases().get(format)
    if uri is None or registry.lookup(uri) is None:
        raise KalfaError(f"export format {format!r} is no export lego; the std ones are onnx, torchscript and "
                         f"state_dict")
    if kalfa_kind(uri) != "export":
        raise KalfaError(f"{uri} is a {kalfa_kind(uri)} lego, not an export")
    models, composites = opened.rebuild()
    name = model or opened.after.get("predicts")
    target = resolve_model(name, models, composites)
    device = build_device(device) if device is not None else Device.cpu()
    device.place(models)
    device.place(composites)
    loader = record_loaders(opened.document, opened.contract)["test"]
    batch = device.move(next(iter(loader)))
    directory = Path(out) if out is not None else Path(run_dir) / "export"
    inputs = traced_inputs(target, batch)
    target.eval()
    with torch.no_grad():
        target(*inputs)
    logger.info(f"exporting {name} ({opened.which} weights) as {uri}")
    path = registry.resolve(uri)(target, inputs, directory, name)
    if path is not None:
        logger.info(f"wrote {path}")
    return Exported(str(path) if path is not None else None, name, uri)


@dataclass
class Generated:
    path: str | None
    samples: torch.Tensor | str


def generate(run_dir, which=None, sets=None, device=None, contract=None) -> Generated:
    opened = open_record(run_dir, which, sets, contract)
    if opened.config.get("generate") is None:
        raise KalfaError(f"{run_dir}: the config has no generate section")
    models, composites = opened.rebuild()
    everything = {**composites, **models}
    for name, state in opened.payload.get("emas", {}).items():
        if name in models:
            ema = Ema(models[name], 1.0)
            ema.load_state_dict(state)
            everything[f"{name}.ema"] = ema
    sampler = opened.store.resolve_params(opened.after["generate"])
    logger.info(f"generating with the {opened.which} models")
    device = build_device(device) if device is not None else Device.cpu()
    device.place(everything)
    seed_all(opened.config.get("seed"))
    samples = sampler(models=everything, prep=opened.prep, rng=device.generator())
    target = Path(run_dir) / "samples"
    write_samples(samples, target)
    path = target / ("samples.txt" if isinstance(samples, str) else "samples.pt")
    return Generated(str(path), samples)


def data_outputs(document, names, contract=None):
    return flow_outputs(document, ("data",), names, contract)


def flow_outputs(document, blocks, names, contract=None):
    flow = {"outputs": list(names)}
    for block in blocks:
        flow[block] = document["flow"][block]
    analysis = analyze({**{key: value for key, value in document.items() if key != "flow"}, "flow": flow}, contract)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gate(analysis.problems)
        pipeline, problems = compile(analysis, [], dry=False)
        gate(problems)
        report = tezgah.run(pipeline, inputs={})
    return report.outputs


__all__ = ["Exported", "Generated", "KalfaError", "Opened", "Plots", "Prepared", "PreparedData", "Prediction",
           "Probe", "RunResult", "check", "export", "generate", "open_record", "plots", "predict", "prepare",
           "prepare_data", "probe", "read_resolved", "resume", "run", "seed_all"]
