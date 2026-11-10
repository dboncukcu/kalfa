import copy
import json
import os
import shlex
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ruamel.yaml.error import YAMLError

from kalfa.commands.sweep.points import LOGS, POINTS_FILE, SWEEP_FILE
from kalfa.commands.sweep.space import is_int
from kalfa.commands.sweep.strategies import strategy_note, strategy_of
from kalfa.core import run as running
from kalfa.core.check import check
from kalfa.core.errors import KalfaError, ProblemsError
from kalfa.core.record import atomic, now
from kalfa.core.spec import parse_value
from kalfa.records import reader


MAPPING_NOTE = "# mapping config: run through the python api"


@dataclass
class Plan:
    paths: tuple
    sets: tuple
    params: tuple
    root: str
    strategy: object
    space: dict
    objective: dict
    points: list | None = None
    config_name: str = "sweep"
    extra: dict = field(default_factory=dict)

    @property
    def count(self):
        return None if self.points is None else len(self.points)

    @property
    def adaptive(self):
        return self.points is None


def objective_note(raw):
    raw = raw if isinstance(raw, dict) else {}
    return {"watch": raw.get("watch"), "mode": raw.get("mode") or "min", "at": raw.get("at") or "best"}


def plan_seed(seed):
    return int(seed) if is_int(seed) else 0


def plan_of(paths=(), sets=(), params=(), mappings=(), record=None):
    paths = tuple(str(path) for path in paths)
    sets = tuple(running.pairs(sets))
    params = tuple(running.pairs(params, "param"))
    mappings = list(mappings)
    checked = check(paths=paths, sets=list(sets), params=list(params), mappings=mappings)
    problems = [problem for problem in checked.problems if record is None or problem.code != "sweep_record"]
    if any(problem.severity == "error" for problem in problems):
        raise ProblemsError(problems)
    raw = checked.tree.settings.get("sweep") if checked.tree is not None else None
    if not isinstance(raw, dict):
        raise KalfaError("the config has no sweep section; write sweep: {strategy, space, objective, record}")
    seed = plan_seed(checked.config.data.get("seed"))
    strategy = strategy_of(raw.get("strategy"))
    if getattr(strategy, "seed", 0) is None:
        strategy.seed = seed
    space = copy.deepcopy(raw["space"])
    plan = Plan(paths, sets, params, str(record) if record is not None else str(raw["record"]), strategy, space,
                objective_note(raw.get("objective")), config_name=Path(paths[0]).stem if paths else "sweep",
                extra={"mappings": mappings, "seed": seed, "strategy": strategy_note(raw.get("strategy"))})
    if callable(getattr(strategy, "points", None)):
        plan.points = list(strategy.points(space, seed))
    return plan


def value_text(value):
    if isinstance(value, str):
        try:
            if parse_value(value) == value:
                return value
        except YAMLError:
            pass
    return json.dumps(value, default=str)


def command_args(plan, index=None, absolute=False, repair=False):
    paths = [os.path.abspath(path) if absolute else path for path in plan.paths] or ["CONFIG"]
    args = ["sweep", *paths]
    for path, value in plan.sets:
        args += ["--set", f"{path}={value_text(value)}"]
    for name, value in plan.params:
        args += ["-p", f"{name}={value_text(value)}"]
    args += ["--record", os.path.abspath(plan.root) if absolute else plan.root]
    if index is not None:
        args += ["--point", str(index)]
    if repair:
        args.append("--repair")
    args.append("--quiet")
    return args


def script_text(plan):
    indices = [None] if plan.adaptive else list(range(plan.count))
    lines = [shlex.join(["kalfa", *command_args(plan, index, absolute=True)]) for index in indices]
    if not plan.paths:
        return "\n".join(["#!/usr/bin/env bash", MAPPING_NOTE, *(f"# {line}" for line in lines)]) + "\n"
    repair = shlex.join(["kalfa", *command_args(plan, absolute=True, repair=True)])
    return "\n".join(["#!/usr/bin/env bash", "set -uo pipefail", "status=0", *(f"{line} || status=1" for line in lines),
                      "# a point that fails leaves the others running; this reruns the failed and missing points:",
                      f"# {repair}", 'exit "$status"']) + "\n"


def condor_arg(text):
    text = str(text).replace('"', '""')
    if any(character.isspace() or character == "'" for character in text):
        return "'" + text.replace("'", "''") + "'"
    return text


def executable():
    found = shutil.which("kalfa")
    return (found, []) if found else (sys.executable, ["-m", "kalfa.cli"])


def submit_text(plan):
    index = None if plan.adaptive else "$(Process)"
    program, prefix = executable()
    arguments = " ".join(condor_arg(arg) for arg in [*prefix, *command_args(plan, index, absolute=True)])
    logs = os.path.join(os.path.abspath(plan.root), LOGS)
    lines = [f"executable = {program}", f'arguments = "{arguments}"', "getenv = True",
             f"output = {logs}/$(Process).out", f"error = {logs}/$(Process).err", f"log = {logs}/sweep.log",
             f"queue {1 if plan.adaptive else plan.count}"]
    if not plan.paths:
        lines = [MAPPING_NOTE, *(f"# {line}" for line in lines)]
    return "\n".join(lines) + "\n"


def write_text(path, text):
    with atomic(path) as temporary:
        temporary.write_text(text, encoding="utf-8")


def points_text(points):
    return "".join(json.dumps({"id": index, "values": values}, default=str) + "\n"
                   for index, values in enumerate(points or []))


def previous_note(root):
    try:
        found = reader.read_json(root, SWEEP_FILE)
    except (OSError, ValueError):
        return {}
    return found if isinstance(found, dict) else {}


def write_plan(plan, log=print):
    root = Path(plan.root)
    root.mkdir(parents=True, exist_ok=True)
    (root / LOGS).mkdir(exist_ok=True)
    previous = previous_note(root)
    note = {"format": 1, "strategy": plan.extra.get("strategy") or {"use": type(plan.strategy).__name__.lower()},
            "space": plan.space, "objective": plan.objective, "points": plan.count}
    if plan.adaptive:
        note["trials"] = getattr(plan.strategy, "trials", None)
    note["started"] = previous.get("started") or now()
    note["config_files"] = [os.path.abspath(path) for path in plan.paths]
    note["record"] = plan.root
    written = [root / SWEEP_FILE, root / POINTS_FILE, root / "sweep.sh", root / "sweep.sub"]
    write_text(written[0], json.dumps(note, indent=1, default=str) + "\n")
    if not plan.adaptive or not written[1].exists():
        write_text(written[1], points_text(plan.points))
    write_text(written[2], script_text(plan))
    written[2].chmod(0o755)
    write_text(written[3], submit_text(plan))
    size = f"{plan.count} points" if not plan.adaptive else f"adaptive, {note['trials']} trials"
    log(f"{plan.root}: {size}; " + ", ".join(path.name for path in written))
    return [str(path) for path in written]
