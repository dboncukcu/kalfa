import logging
from pathlib import Path

from kalfa.core.errors import KalfaError
from kalfa.core.record import STOP_FILE, RecordWriter, now
from kalfa.records import reader
from kalfa.records import sweep as sweeping


STOPPABLE = ("running", "stale")
LOGGER = logging.getLogger("kalfa.commands")


def kind_of(record):
    directory = Path(record)
    if reader.is_record(directory):
        return "record"
    if sweeping.is_sweep(directory):
        return "sweep"
    raise KalfaError(f"{record} is neither a record directory nor a sweep root")


def write_stop(directory, by):
    writer = RecordWriter(directory)
    writer.write_json(STOP_FILE, {"by": by, "at": now()})
    return str(writer.path(STOP_FILE))


def state_in(directory):
    try:
        return reader.state_of(reader.read_manifest(directory))
    except (OSError, ValueError):
        return "unreadable"


def stoppable(root):
    return [point for point in sweeping.point_dirs(root) if state_in(point) in STOPPABLE]


def stop_record(record, by="cli"):
    directory = Path(record)
    if kind_of(directory) == "record":
        state = state_in(directory)
        if state not in STOPPABLE:
            LOGGER.warning("%s is %s; nothing to stop", record, state)
            return []
        return [write_stop(directory, by)]
    points = stoppable(directory)
    if not points:
        LOGGER.warning("nothing to stop under %s: no running or stale points", record)
    return [write_stop(point, by) for point in points]


def stop_many(records, by="cli"):
    for record in records:
        kind_of(record)
    written = []
    for record in records:
        written.extend(stop_record(record, by=by))
    return written
