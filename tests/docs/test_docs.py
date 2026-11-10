import json
import sys
from pathlib import Path

import pytest
from ruamel.yaml import YAML

import kalfa
from helpers import housing_frame
from kalfa.cli import HANDLERS
from kalfa.core.aliases import ALIASES
from kalfa.records.reader import read_lines


ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
DOCS = ROOT / "DOCS.md"
EXAMPLES = ROOT / "examples" / "README.md"
DOCUMENTS = (README, DOCS, EXAMPLES)
DASHES = (chr(0x2014), chr(0x2013), " - ")
REFERENCE = "## Reference config"
PIECES = "## Writing your own pieces"
TOGETHER = "### Putting the pieces together"
DATA_CLASS = "### A data class"


def text_of(path):
    return path.read_text(encoding="utf-8")


def prose_lines(text):
    inside = False
    for number, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("```"):
            inside = not inside
            continue
        if not inside:
            yield number, line


def fenced(text):
    found = []
    language = None
    lines = None
    for line in text.splitlines():
        stripped = line.strip()
        if lines is None:
            if stripped.startswith("```"):
                language = stripped[3:].strip()
                lines = []
        elif stripped.startswith("```"):
            found.append((language, "\n".join(lines) + "\n"))
            lines = None
        else:
            lines.append(line)
    assert lines is None, "a fenced block is not closed"
    return found


def section(text, heading):
    lines = text.splitlines()
    assert heading in lines, f"no heading {heading!r}"
    start = lines.index(heading) + 1
    level = len(heading) - len(heading.lstrip("#"))
    end = len(lines)
    inside = False
    for position in range(start, len(lines)):
        line = lines[position]
        if line.lstrip().startswith("```"):
            inside = not inside
        elif not inside and line.startswith("#") and len(line) - len(line.lstrip("#")) <= level:
            end = position
            break
    return "\n".join(lines[start:end]) + "\n"


def first(text, language):
    for found, body in fenced(text):
        if found == language:
            return body
    raise AssertionError(f"no {language} block")


def home(directory, text, name="config.yaml"):
    directory.mkdir(parents=True, exist_ok=True)
    housing_frame().to_parquet(directory / "housing.parquet", index=False)
    config = directory / name
    config.write_text(text, encoding="utf-8")
    return config


def turns_of(record):
    return [line["turn"] for line in read_lines(Path(record) / "history.jsonl")]


def status_of(record):
    return json.loads((Path(record) / "manifest.json").read_text(encoding="utf-8"))["status"]


@pytest.fixture
def pieces_module():
    saved = dict(ALIASES)
    sys.modules.pop("pieces", None)
    yield
    sys.modules.pop("pieces", None)
    ALIASES.clear()
    ALIASES.update(saved)


@pytest.mark.parametrize("path", DOCUMENTS, ids=lambda path: path.relative_to(ROOT).as_posix())
def test_prose_has_no_dash_punctuation(path):
    bad = []
    for number, line in prose_lines(text_of(path)):
        if any(mark in line for mark in DASHES) or line.lstrip().startswith("- "):
            bad.append(f"{path.name}:{number}: {line}")
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("path", DOCUMENTS, ids=lambda path: path.relative_to(ROOT).as_posix())
def test_yaml_blocks_parse_and_python_blocks_compile(path):
    for language, body in fenced(text_of(path)):
        if language == "yaml":
            YAML(typ="safe").load(body)
        elif language == "python":
            compile(body, str(path), "exec")


def test_readme_config_passes_check(tmp_path):
    config = home(tmp_path, first(text_of(README), "yaml"))
    checked = kalfa.check(str(config))
    assert checked.ok, checked.text()


def test_readme_python_snippet_trains(tmp_path, monkeypatch):
    code = first(text_of(README), "python")
    assert code.count("epochs=20") == 1 and code.count("kalfa.Run(") == 1
    code = code.replace("epochs=20", "epochs=1").replace("kalfa.Run(", 'kalfa.Run(device="cpu", ')
    home(tmp_path, "")
    monkeypatch.chdir(tmp_path)
    exec(compile(code, str(README), "exec"), {"__name__": "readme"})
    records = [path for path in (tmp_path / "runs").iterdir() if (path / "manifest.json").is_file()]
    assert len(records) == 1 and status_of(records[0]) == "done"


def test_reference_config_passes_check_and_trains_two_turns(tmp_path):
    config = home(tmp_path, first(section(text_of(DOCS), REFERENCE), "yaml"))
    checked = kalfa.check(str(config))
    assert checked.ok, checked.text()
    assert not checked.warnings, checked.text()
    record = tmp_path / "record"
    kalfa.run(str(config), sets=["device=cpu"], params=["epochs=2"], record=str(record))
    assert status_of(record) == "done"
    assert turns_of(record) == [1, 2]
    patched = {event["path"] for event in read_lines(record / "events.jsonl") if event["kind"] == "patch"}
    assert patched
    for name in ("plan.json", "resolved.yaml", "predictions.parquet", "checkpoints/best.pt", "checkpoints/final.pt",
                 "model_summary.json", "fitted/fields.json"):
        assert (record / name).is_file(), name


def test_the_pieces_of_the_docs_train_together(tmp_path, pieces_module):
    pieces = section(text_of(DOCS), PIECES)
    code = "\n\n".join(body for language, body in fenced(pieces) if language == "python")
    together = home(tmp_path / "together", first(section(pieces, TOGETHER), "yaml"))
    (together.parent / "pieces.py").write_text(code, encoding="utf-8")
    checked = kalfa.check(str(together))
    assert checked.ok, checked.text()
    record = tmp_path / "together_record"
    kalfa.run(str(together), sets=["device=cpu", "train.epochs=2"], record=str(record))
    assert status_of(record) == "done" and turns_of(record) == [1, 2]
    for name in ("predictions.parquet", "samples/samples.pt", "plots/residuals.png", "snapshots.jsonl",
                 "best_seen.json"):
        assert (record / name).is_file(), name


def test_the_data_class_of_the_docs_trains(tmp_path, pieces_module):
    pieces = section(text_of(DOCS), PIECES)
    code = "\n\n".join(body for language, body in fenced(pieces) if language == "python")
    config = home(tmp_path / "sine", first(section(pieces, DATA_CLASS), "yaml"))
    (config.parent / "pieces.py").write_text(code, encoding="utf-8")
    checked = kalfa.check(str(config))
    assert checked.ok, checked.text()
    record = tmp_path / "sine_record"
    kalfa.run(str(config), sets=["device=cpu", "train.epochs=2"], record=str(record))
    assert status_of(record) == "done" and turns_of(record) == [1, 2]


def test_docs_name_every_command():
    text = text_of(DOCS)
    missing = [name for name in HANDLERS if f"kalfa {name}" not in text]
    assert not missing, missing
