from pathlib import Path

from ..core import run as running


def signatures(directory):
    if not directory.is_dir():
        return {}
    found = {}
    for path in directory.rglob("*"):
        if path.is_file() and not path.name.startswith("."):
            stat = path.stat()
            found[path] = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
    return found


def draw(record, only=None, which=None, device=None):
    ctx = running.open_record(record, which=which, device=device)
    return running.draw_plots(ctx, record, only)


def redraw(record, only=None, which=None, device=None):
    directory = Path(record) / "plots"
    before = signatures(directory)
    draw(record, only, which, device)
    after = signatures(directory)
    return [str(path) for path in sorted(after) if before.get(path) != after[path]]
