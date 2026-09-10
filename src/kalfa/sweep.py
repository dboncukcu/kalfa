import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from cirak.registry import registry

from .api import gate, prepare_data, run
from .config import load_surface, parse_sets, resolve_alias
from .kinds import kalfa_kind
from .record import Record
from .schema import Schema
from .std.common.history import History
from .std.strategy.base import Strategy, parse_space



class SweepError(ValueError):
    pass


@dataclass
class Plan:
    strategy: Strategy
    uri: str
    space: dict
    objective: dict
    root: Path
    total: int
    raw: dict = field(default_factory=dict)
    paths: list = field(default_factory=list)

    @property
    def deterministic(self):
        return bool(self.strategy.deterministic)


def strategy_of(section, aliases):
    call = section.get("strategy")
    if isinstance(call, str):
        uri, params = call, {}
    elif isinstance(call, dict) and isinstance(call.get("uri"), str):
        uri, params = call["uri"], dict(call.get("params") or {})
    else:
        raise SweepError("sweep.strategy must be a strategy lego: a short name or {uri, params}")
    resolved = resolve_alias(uri, aliases)
    if resolved is None or registry.lookup(resolved) is None:
        raise SweepError(f"sweep.strategy {uri!r} is not a known strategy lego")
    if kalfa_kind(resolved) != "strategy":
        raise SweepError(f"{resolved} is a {kalfa_kind(resolved)} lego, sweep.strategy needs a strategy")
    return registry.resolve(resolved)(**params), resolved


def plan(paths, sets=None, record=None) -> Plan:
    surface = load_surface(paths, sets)
    gate(surface.problems)
    section = surface.data.get("sweep")
    if not isinstance(section, dict):
        raise SweepError("the config has no sweep section")
    strategy, uri = strategy_of(section, surface.aliases)
    space = parse_space(section.get("space"))
    objective = dict(section.get("objective") or {})
    root = Path(record if record is not None else section.get("record") or "runs/sweep")
    return Plan(strategy, uri, space, objective, root, int(strategy.total(space)), surface.raw, list(paths))


def swept_in_data(raw, space):
    found = []

    def walk(value):
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str):
            for name in space:
                if f"${name}$" in value and name not in found:
                    found.append(name)

    walk((raw or {}).get("data"))
    return found


def submit_text(plan):
    config = Path(plan.paths[0]).resolve() if plan.paths else Path("config.yaml").resolve()
    return "\n".join([
        "# kalfa sweep: written once by kalfa sweep --plan, edited for the site, never overwritten",
        "include : sweep.plan",
        "executable            = sweep.sh",
        "arguments             = $(Process)",
        f"initialdir            = {config.parent}",
        f"transfer_input_files  = {', '.join(Path(path).name for path in plan.paths)}",
        "request_gpus          = 1",
        "+MaxRuntime           = 43200",
        "output                = $(ROOT)/condor/$(Process).out",
        "error                 = $(ROOT)/condor/$(Process).err",
        "log                   = $(ROOT)/condor/sweep.log",
        "queue $(N)",
        "",
    ])


def script_text(plan):
    return "\n".join([
        "#!/usr/bin/env bash",
        "# kalfa sweep: activate the environment of the site, then run one point; edited freely, never overwritten",
        "set -euo pipefail",
        "# source /path/to/venv/bin/activate",
        f"CONFIG=\"{' '.join(str(Path(path).resolve()) for path in plan.paths)}\"",
        f"ROOT=\"{plan.root.resolve()}\"",
        "kalfa sweep $CONFIG --id \"$1\" --record \"$ROOT\" --no-progress --log info",
        "",
    ])


def plan_text(plan):
    return "\n".join([f"N = {plan.total}", f"CONFIG = {' '.join(str(Path(path).resolve()) for path in plan.paths)}",
                       f"ROOT = {plan.root.resolve()}", ""])


def write_plan(plan, sets=None, params=None, prepare=False, contract=None, log=print):
    root = Record(plan.root)
    root.directory.mkdir(parents=True, exist_ok=True)
    existing = root.read_json("manifest.json") or {}
    prepared = bool(existing.get("prepared"))
    if prepare:
        swept = swept_in_data(plan.raw, plan.space)
        if swept:
            raise SweepError(f"the data section reads the swept {swept}, so the data differs per point; "
                             f"--prepare-data cannot prepare it once")
        prepare_data(plan.paths, parse_sets(sets, params), out=root.path("data"), contract=contract)
        prepared = True
    note = {"kind": "sweep", "started": existing.get("started") or root.manifest("sweep")["started"],
            "strategy": plan.uri, "space": {name: choices_text(value) for name, value in plan.space.items()},
            "objective": {key: plan.objective.get(key) for key in Schema.objective}, "total": plan.total,
            "config": [str(Path(path).resolve()) for path in plan.paths], "prepared": prepared}
    root.write_json("manifest.json", {**root.manifest("sweep"), **note})
    root.write_text("sweep.plan", plan_text(plan))
    written = ["manifest.json", "sweep.plan"]
    for name, text in (("sweep.sub", submit_text(plan)), ("sweep.sh", script_text(plan))):
        if root.path(name).exists():
            continue
        root.write_text(name, text)
        written.append(name)
    root.path("sweep.sh").chmod(0o755)
    log(f"{plan.root}: {', '.join(written)}" + (" (the data prepared under data/)" if prepared else ""))
    return root.read_json("manifest.json")


def choices_text(value):
    return list(value) if isinstance(value, (list, tuple)) else repr(value)


def prepared_dir(plan):
    manifest = Record(plan.root).read_json("manifest.json") or {}
    folder = plan.root / "data"
    return folder if manifest.get("prepared") and (folder / "manifest.json").exists() else None


def point_dir(root, index):
    return Path(root) / f"{int(index):04d}"


def yaml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = repr(value)
        mantissa, _, exponent = text.partition("e")
        if exponent and "." not in mantissa:
            text = f"{mantissa}.0e{exponent}"
        return text
    return json.dumps(value)


def point_params(point):
    return [f"{name}={yaml_value(value)}" for name, value in point.items()]


def point_of(plan, index):
    if not plan.deterministic:
        raise SweepError(f"{plan.uri} proposes points from the objectives fed back; it runs in the local loop "
                         f"(kalfa sweep cfg.yaml) and takes no --id or --show")
    return plan.strategy.point(plan.space, int(index))


def objective_of(record, objective):
    monitor = objective.get("monitor")
    try:
        return History.read(record).best(monitor, objective.get("mode", "min"), objective.get("at", "best"))
    except ValueError:
        raise SweepError(f"{record}: history has no value for objective monitor {monitor!r}") from None


def write_point(record, plan, index, point, value, turn):
    entry = {"id": int(index), "point": point, "strategy": plan.uri, "total": plan.total,
             "objective": {**{key: plan.objective.get(key) for key in Schema.objective}, "value": value, "turn": turn},
             "record": str(record)}
    (Path(record) / "sweep.json").write_text(json.dumps(entry, indent=2))
    return entry


def read_point(record):
    path = Path(record) / "sweep.json"
    return json.loads(path.read_text()) if path.exists() else None


def run_point(paths, sets, params, plan, index, point=None, when=None):
    if point is None:
        point = point_of(plan, index)
    record = point_dir(plan.root, index)
    layer = parse_sets([*(sets or []), f"record={record}"], [*(params or []), *point_params(point)])
    run(paths, layer, when=when, prepared=prepared_dir(plan),
        identity={"kind": "point", "id": int(index), "values": dict(point), "root": str(plan.root)})
    value, turn = objective_of(record, plan.objective)
    return write_point(record, plan, index, point, value, turn)


def child_command(paths, sets, params, root, index, point=None):
    command = [sys.executable, "-m", "kalfa.cli", "sweep", *paths, "--id", str(index), "--record", str(root)]
    if point is not None:
        command += ["--point", json.dumps(point)]
    for text in sets or []:
        command += ["--set", text]
    for text in params or []:
        command += ["-p", text]
    return command


def local_loop(paths, sets, params, plan, log=print):
    write_plan(plan, sets, params, log=log)
    entries = []
    for index in range(plan.total):
        record = point_dir(plan.root, index)
        done = read_point(record)
        if done is not None:
            log(f"{record}: done, {done['objective']['monitor']}={done['objective']['value']:.6g}")
            entries.append(done)
            continue
        if record.exists():
            log(f"{record}: exists without sweep.json (unfinished or failed), skipped; remove it to rerun")
            entries.append(None)
            continue
        trial = point = None
        if not plan.deterministic:
            trial, point = plan.strategy.ask(plan.space, plan.objective.get("mode", "min"))
        completed = subprocess.run(child_command(paths, sets, params, plan.root, index, point))
        entry = read_point(record) if completed.returncode == 0 else None
        if entry is None:
            log(f"{record}: failed (exit {completed.returncode})")
        else:
            log(f"{record}: {entry['point']} -> {entry['objective']['monitor']}={entry['objective']['value']:.6g}")
        if trial is not None:
            plan.strategy.tell(trial, entry["objective"]["value"] if entry else None)
        entries.append(entry)
    return entries
