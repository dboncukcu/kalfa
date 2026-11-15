import logging
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from kalfa.commands.sweep.plan import command_args, write_plan
from kalfa.commands.sweep.points import (
    LOGS,
    OUTSIDE_STOPS,
    SWEEP_FILE,
    clear_point,
    manifest_of,
    planned_values,
    point_dir,
    recorded_points,
    stopped_by_rule,
)
from kalfa.commands.sweep.space import is_int
from kalfa.core import run as running
from kalfa.records import reader


LOGGER = logging.getLogger("kalfa.commands")


def silent(text):
    return None


def say_of(monitor):
    say = getattr(monitor, "say", None)
    return say if callable(say) else silent


def run_point(plan, index, values=None, monitor=None):
    index = int(index)
    if values is None:
        values = planned_values(plan, index)
    if not (Path(plan.root) / SWEEP_FILE).exists():
        write_plan(plan, log=say_of(monitor))
    directory = Path(point_dir(plan.root, index))
    clear_point(directory, index)
    return running.run(paths=plan.paths, sets=[*plan.sets, ("sweep.record", plan.root)],
                       params=[*plan.params, *values.items()], mappings=plan.extra.get("mappings", ()),
                       record=str(directory), monitor=monitor,
                       sweep={"root": plan.root, "id": index, "values": values})


def values_text(values):
    return "  ".join(f"{key}={value}" for key, value in values.items())


def interrupted(result):
    stopping = getattr(getattr(result, "context", None), "stopping", None)
    return isinstance(stopping, str) and stopping.startswith(OUTSIDE_STOPS)


def child(plan, index, logs, halt):
    if halt.is_set():
        return index, None
    command = [sys.executable, "-m", "kalfa.cli", *command_args(plan, index)]
    with open(logs / f"{index:04d}.log", "w", encoding="utf-8") as out:
        code = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT, check=False).returncode
    directory = Path(point_dir(plan.root, index))
    manifest = manifest_of(directory)
    if code == 0 and manifest is not None and manifest.get("status") == "stopped" and not stopped_by_rule(directory):
        halt.set()
    return index, code


def in_children(plan, indices, jobs, log):
    logs = Path(plan.root) / LOGS
    logs.mkdir(parents=True, exist_ok=True)
    halt = threading.Event()
    results, skipped = [], 0
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for index, code in pool.map(lambda index: child(plan, index, logs, halt), indices):
            directory = point_dir(plan.root, index)
            output = logs / f"{index:04d}.log"
            if code is None:
                skipped += 1
            elif code == 0:
                results.append(running.Result(record=directory, history=reader.read_history(directory), context=None))
            else:
                LOGGER.warning("point %04d failed (exit %s); its output is in %s", index, code, output)
                results.append(None)
    if skipped:
        log(f"a point was stopped from outside; {skipped} points did not start and --repair runs them")
    return results


def resumed(plan):
    resume = getattr(plan.strategy, "resume", None)
    if not callable(resume):
        return None
    points = recorded_points(plan)
    told = resume(plan.space, plan.objective, points)
    return told if is_int(told) else len(points)
