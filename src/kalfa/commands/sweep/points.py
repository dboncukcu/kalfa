import json
import shutil
from pathlib import Path

from kalfa.commands.sweep.space import is_int
from kalfa.core.errors import KalfaError
from kalfa.core.record import STOP_FILE
from kalfa.records import reader
from kalfa.records.sweep import objective_of, point_dirs


REPAIRED = ("failed", "stopped")
OUTSIDE_STOPS = ("keyboard", "stop.json")
SWEEP_FILE = "sweep.json"
POINTS_FILE = "points.jsonl"
LOGS = "logs"


def point_dir(root, index):
    return f"{root}/{index:04d}"


def manifest_of(directory):
    try:
        found = reader.read_manifest(directory)
    except (OSError, ValueError):
        return None
    return found if isinstance(found, dict) else None


def stopped_by_rule(directory):
    ends = [line for line in reader.read_lines(Path(directory) / "events.jsonl") if line.get("kind") == "run_end"]
    stopping = ends[-1].get("stopping") if ends else None
    return isinstance(stopping, str) and not stopping.startswith(OUTSIDE_STOPS)


def complete(directory, manifest=None):
    manifest = manifest if manifest is not None else manifest_of(directory)
    if manifest is None:
        return False
    status = manifest.get("status")
    return status == "done" or (status == "stopped" and stopped_by_rule(directory))


def abandoned(directory, manifest):
    return reader.state_of(manifest) == "stale" and (Path(directory) / STOP_FILE).is_file()


def repairable(directory):
    manifest = manifest_of(directory)
    if manifest is None:
        return True
    if manifest.get("status") == "running":
        return abandoned(directory, manifest)
    return manifest.get("status") in REPAIRED and not complete(directory, manifest)


def clear_point(directory, index):
    if not directory.is_dir() or not any(directory.iterdir()):
        return
    manifest = manifest_of(directory)
    if manifest is not None and complete(directory, manifest):
        raise KalfaError(f"point {index} is {manifest.get('status')} ({directory}); --repair runs only the failed "
                         f"and missing points")
    if manifest is not None and manifest.get("status") == "running" and not abandoned(directory, manifest):
        if reader.state_of(manifest) == "stale":
            raise KalfaError(f"point {index} was running and its heartbeat is stale ({directory}); if its process is "
                             f"dead, kalfa stop {directory} marks it and --repair runs it again, or remove the "
                             f"directory")
        raise KalfaError(f"point {index} is running ({directory}); kalfa stop {directory} asks it to stop after its "
                         f"turn")
    shutil.rmtree(directory)


def recorded_values(plan):
    found = {}
    for directory in point_dirs(plan.root):
        manifest = manifest_of(directory)
        sweep = manifest.get("sweep") if manifest is not None else None
        if isinstance(sweep, dict) and isinstance(sweep.get("values"), dict):
            found[int(directory.name)] = dict(sweep["values"])
    for line in reader.read_lines(Path(plan.root) / POINTS_FILE):
        if is_int(line.get("id")) and isinstance(line.get("values"), dict):
            found[line["id"]] = dict(line["values"])
    return found


def planned_values(plan, index):
    if plan.points is not None:
        if not 0 <= index < len(plan.points):
            raise KalfaError(f"point {index} is not in the plan of {plan.root}, which has {len(plan.points)} points")
        return plan.points[index]
    found = recorded_values(plan).get(index)
    if found is None:
        raise KalfaError(f"{plan.root}: the strategy is adaptive and point {index} was never asked; run the sweep "
                         f"without --point")
    return found


def next_index(plan):
    known = [int(directory.name) for directory in point_dirs(plan.root)]
    known += [line["id"] for line in reader.read_lines(Path(plan.root) / POINTS_FILE) if is_int(line.get("id"))]
    return max(known) + 1 if known else 0


def append_point(root, index, values):
    with open(Path(root) / POINTS_FILE, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"id": index, "values": values}, default=str) + "\n")


def recorded_points(plan):
    points = []
    for index, values in sorted(recorded_values(plan).items()):
        directory = Path(point_dir(plan.root, index))
        summary = reader.summary_of(directory) if directory.is_dir() else None
        failed = not isinstance(summary, dict) or summary.get("status") == "failed"
        points.append((values, None if failed else objective_of(summary, plan.objective)))
    return points


def failed_points(plan):
    indices = range(len(plan.points)) if plan.points is not None else sorted(recorded_values(plan))
    return [index for index in indices if repairable(Path(point_dir(plan.root, index)))]
