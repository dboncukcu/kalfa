"""The record directory: naming, resolved.yaml and flow.yaml, and reading a run back."""

import json
from datetime import datetime
from io import StringIO
from pathlib import Path

from cirak.api import annotated
from ruamel.yaml import YAML

from .config import written_config

DATETIME_TOKEN = "$datetime$"


def stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def record_dir(config, when=None):
    """The record directory of a config; $datetime$ is filled once, here."""
    record = config.get("record")
    if not isinstance(record, str):
        raise ValueError("record must be a path")
    return record.replace(DATETIME_TOKEN, when or stamp())


def resolved_text(surface) -> str:
    yaml = YAML()
    yaml.width = 120
    stream = StringIO()
    yaml.dump(annotated(written_config(surface.data), surface.overrides), stream)
    return stream.getvalue()


def write_resolved(directory, surface):
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    (target / "resolved.yaml").write_text(resolved_text(surface))


def write_flow(directory, text):
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    (target / "flow.yaml").write_text(text)


def read_resolved(directory):
    return YAML(typ="safe").load((Path(directory) / "resolved.yaml").read_text())


def read_history(directory):
    path = Path(directory) / "history.jsonl"
    if not path.exists():
        return []
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            lines.append(json.loads(line))
    return lines


def is_run(directory):
    return (Path(directory) / "resolved.yaml").exists()


def resume_source(directory):
    """The checkpoint a resume starts from: last.pt, else final/state.pt of a finished run, else None."""
    last = Path(directory) / "checkpoints" / "last.pt"
    if last.exists():
        return last
    final = Path(directory) / "final" / "state.pt"
    if final.exists():
        return final
    return None


def write_resume_note(directory, source_run, checkpoint):
    note = {"resume_from": str(source_run), "checkpoint": str(checkpoint)}
    (Path(directory) / "resume.json").write_text(json.dumps(note, indent=2))


def resume_chain(directory):
    """The runs a record continues, nearest first."""
    chain = []
    current = Path(directory)
    while (current / "resume.json").exists():
        note = json.loads((current / "resume.json").read_text())
        current = Path(note["resume_from"])
        chain.append(str(current))
        if len(chain) > 100:
            break
    return chain
