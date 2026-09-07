"""kalfa sweep: the points of a search space run as ordinary records under one root.

Every point is a plain run whose record is ``<root>/<id>``, written together with ``sweep.json`` (the id, the
point, the strategy, the total and the objective). Strategies deterministic by id run anywhere with ``--id``; a fed
back strategy proposes points in the local loop, where every point runs in a subprocess and its objective is read
back from history.jsonl.
"""

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from cirak.registry import registry

from .config import load_surface, parse_sets, resolve_alias
from .kinds import kalfa_kind
from .record import read_history
from .std.strategy import parse_space

SWEEP_KEYS = ("strategy", "space", "objective", "record")
OBJECTIVE_KEYS = ("monitor", "mode", "at")


class SweepError(ValueError):
    pass


@dataclass
class Plan:
    strategy: object
    uri: str
    space: dict
    objective: dict
    root: Path
    total: int

    @property
    def deterministic(self):
        return bool(getattr(self.strategy, "deterministic", False))


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
    """The sweep of a config: the strategy built, the space parsed, the objective and the root."""
    from .api import gate

    surface = load_surface(paths, sets)
    gate(surface.problems)
    section = surface.data.get("sweep")
    if not isinstance(section, dict):
        raise SweepError("the config has no sweep section")
    strategy, uri = strategy_of(section, surface.aliases)
    space = parse_space(section.get("space"))
    objective = dict(section.get("objective") or {})
    root = Path(record if record is not None else section.get("record") or "runs/sweep")
    return Plan(strategy, uri, space, objective, root, int(strategy.total(space)))


def point_dir(root, index):
    return Path(root) / f"{int(index):04d}"


def yaml_value(value):
    """A point value as a -p value: YAML that reads back as the same scalar."""
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
    """(value, turn) of the objective in a record's history: the best turn by mode, or the last turn."""
    monitor = objective.get("monitor")
    mode = objective.get("mode", "min")
    at = objective.get("at", "best")
    lines = [line for line in read_history(record)
             if isinstance(line.get(monitor), (int, float)) and line[monitor] == line[monitor]]
    if not lines:
        raise SweepError(f"{record}: history has no value for objective monitor {monitor!r}")
    if at == "last":
        line = lines[-1]
    else:
        line = (min if mode == "min" else max)(lines, key=lambda entry: entry[monitor])
    return float(line[monitor]), int(line.get("turn", len(lines)))


def write_point(record, plan, index, point, value, turn):
    entry = {"id": int(index), "point": point, "strategy": plan.uri, "total": plan.total,
             "objective": {**{key: plan.objective.get(key) for key in OBJECTIVE_KEYS}, "value": value, "turn": turn},
             "record": str(record)}
    (Path(record) / "sweep.json").write_text(json.dumps(entry, indent=2))
    return entry


def read_point(record):
    path = Path(record) / "sweep.json"
    return json.loads(path.read_text()) if path.exists() else None


def run_point(paths, sets, params, plan, index, point=None, when=None):
    """Run one point in this process: the record is <root>/<id>, sweep.json is written after the run."""
    from .api import run

    if point is None:
        point = point_of(plan, index)
    record = point_dir(plan.root, index)
    layer = parse_sets([*(sets or []), f"record={record}"], [*(params or []), *point_params(point)])
    run(paths, layer, when=when)
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
    """Every point in a subprocess; finished points (sweep.json) are kept, other existing directories are skipped;
    a fed back strategy learns every objective before proposing the next point."""
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
