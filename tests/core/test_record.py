import importlib.util
import json
import shutil
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from ruamel.yaml import YAML

from kalfa.core.device import Device
from kalfa.core.errors import RunError
from kalfa.core.record import RecordWriter, copy_plugins, git_note, host_note, resolved_text, start_record


DATA = {
    "params": {"epochs": 3, "lr": 0.001},
    "seed": 7,
    "data": {"use": "table", "file": "housing.parquet", "split": [0.7, 0.15, 0.15],
             "features": {"x*": "standard_scaler"}, "targets": {"price": "standard_scaler"}, "batch": 128},
    "models": {"net": {"use": "mlp", "hidden": [128, 128], "out": 1}},
    "losses": {"mse": "mse", "huber": {"use": "torch.nn.HuberLoss", "delta": 1.0}},
    "optimizers": {"main": {"use": "adam", "lr": 0.001, "over": "@models.net", "minimizes": "@losses.mse"}},
    "train": {"epochs": 3, "checkpoint": {"watch": "val/rmse"}, "report": "best", "evaluate": None},
    "rules": {"to_huber": {"when": {"use": "after_epoch", "at": 50},
                           "set": {"optimizers.main.minimizes": "@losses.huber"}}},
    "record": "runs/housing_20261107_120000",
}


def make_config(overrides=None, directories=(), plugins=()):
    return SimpleNamespace(data=json.loads(json.dumps(DATA)), params=dict(DATA["params"]), plugins=list(plugins),
                           directories=list(directories), overrides=dict(overrides or {}))


def load_module(path, name, monkeypatch, package=False):
    locations = [str(Path(path).parent)] if package else None
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=locations)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def hidden_files(directory):
    return [path.name for path in Path(directory).iterdir() if path.name.startswith(".")]


def test_write_json_is_atomic(tmp_path):
    writer = RecordWriter(tmp_path / "run")
    writer.write_json("note.json", {"x": 1})
    assert writer.read_json("note.json") == {"x": 1}
    writer.write_json("note.json", {"x": 2})
    assert json.loads((tmp_path / "run" / "note.json").read_text()) == {"x": 2}
    writer.write_text("resolved.yaml", "a: 1\n")
    assert (tmp_path / "run" / "resolved.yaml").read_text() == "a: 1\n"
    assert hidden_files(tmp_path / "run") == []
    assert writer.read_json("missing.json") is None
    writer.append("history.jsonl", {"turn": 1})
    mode = stat.S_IMODE((tmp_path / "run" / "history.jsonl").stat().st_mode)
    assert stat.S_IMODE((tmp_path / "run" / "note.json").stat().st_mode) == mode
    assert stat.S_IMODE((tmp_path / "run" / "resolved.yaml").stat().st_mode) == mode


def test_manifest_merges_and_stamps(tmp_path):
    writer = RecordWriter(tmp_path)
    first = writer.manifest(format=1, status="running", name="x")
    assert first["status"] == "running" and "updated" in first
    assert writer.read_json("manifest.json") == first
    later = writer.manifest(status="done", finished="soon")
    assert later["format"] == 1 and later["name"] == "x" and later["status"] == "done"
    assert later["finished"] == "soon" and "updated" in later
    assert writer.read_json("manifest.json") == later
    assert hidden_files(tmp_path) == []


def test_append_and_event(tmp_path):
    writer = RecordWriter(tmp_path / "deep" / "run")
    writer.append("history.jsonl", {"turn": 1, "val/rmse": 0.5})
    writer.append("history.jsonl", {"turn": 2, "val/rmse": 0.4})
    lines = [json.loads(line) for line in writer.path("history.jsonl").read_text().splitlines()]
    assert [line["turn"] for line in lines] == [1, 2]
    writer.event("run_start", turn=0, reason="test")
    event = json.loads(writer.path("events.jsonl").read_text().splitlines()[0])
    assert event["kind"] == "run_start" and event["turn"] == 0 and event["reason"] == "test" and "at" in event


def test_stop_note(tmp_path):
    writer = RecordWriter(tmp_path)
    assert writer.stop_requested() is False and writer.stop_note() is None
    writer.write_json("stop.json", {"by": "board"})
    assert writer.stop_requested() is True and writer.stop_note() == {"by": "board"}


def test_resolved_text_round_trip_with_override_comments():
    overrides = {"train.epochs": "--set", "data.split[1]": "cli", "models.net": "02.yaml",
                 "losses.huber.delta": "fold.yaml", "missing.path": "nowhere", "data.split[9]": "nowhere"}
    text = resolved_text(make_config(overrides))
    assert "nowhere" not in text
    marked = {label: [line for line in text.splitlines() if f"# {label}" in line]
              for label in ("--set", "cli", "02.yaml", "fold.yaml")}
    assert all(len(lines) == 1 for lines in marked.values()), text
    assert "epochs: 3" in marked["--set"][0]
    assert "0.15" in marked["cli"][0]
    assert "net:" in marked["02.yaml"][0]
    assert "delta: 1.0" in marked["fold.yaml"][0]
    assert YAML(typ="safe").load(text) == DATA


def test_resolved_text_without_overrides_has_no_comments():
    text = resolved_text(make_config())
    assert "#" not in text
    assert YAML(typ="safe").load(text) == DATA
    assert text.startswith("params:")


def test_host_note_keys():
    note = host_note()
    assert set(note) == {"hostname", "platform", "python", "torch", "cuda", "cwd"}
    assert note["python"].startswith("3.") and note["cwd"] == str(Path.cwd())
    assert note["cuda"] is None or isinstance(note["cuda"], str)


def test_git_note_on_repository_and_plain_directory(tmp_path):
    root = Path(__file__).resolve().parents[2]
    assert git_note(tmp_path) is None
    if shutil.which("git") is None or not (root / ".git").exists():
        pytest.skip("no git repository around the tests")
    note = git_note(root)
    assert set(note) == {"commit", "dirty"}
    assert len(note["commit"]) == 40 and all(char in "0123456789abcdef" for char in note["commit"])
    assert isinstance(note["dirty"], bool)


def test_start_record_writes_the_record(tmp_path, monkeypatch):
    legos = tmp_path / "legos"
    legos.mkdir()
    source = legos / "kalfa_record_legos.py"
    source.write_text("def log_cosh(pred, target):\n    return pred\n")
    load_module(source, "kalfa_record_legos", monkeypatch)
    config = make_config({"train.epochs": "--set"}, directories=[legos])
    ctx = SimpleNamespace(config=config, tree=SimpleNamespace(imports=["kalfa_record_legos.log_cosh",
                                                                        "torch.nn.HuberLoss"]),
                          train=SimpleNamespace(unit="epochs", total_turns=lambda: 3), device=Device("cpu"),
                          record=None)
    config_file = tmp_path / "config.yaml"
    config_file.write_text("seed: 7\n")
    directory = tmp_path / "runs" / "housing"
    writer = start_record(ctx, directory, paths=[config_file], plan={"format": 1, "nodes": {}})
    assert ctx.record is writer and writer.directory == directory
    manifest = writer.read_json("manifest.json")
    assert manifest["format"] == 1 and manifest["kalfa"].startswith("3.") and manifest["name"] == "housing"
    assert manifest["status"] == "running" and manifest["unit"] == "epochs" and manifest["turns_planned"] == 3
    assert manifest["params"] == {"epochs": 3, "lr": 0.001}
    assert manifest["config_files"] == [str(config_file.resolve())]
    assert manifest["device"] == "cpu" and manifest["sweep"] is None and manifest["failure"] is None
    assert manifest["resumed_from"] is None and manifest["finished"] is None
    assert "started" in manifest and "updated" in manifest and "git" in manifest
    assert set(manifest["host"]) == {"hostname", "platform", "python", "torch", "cuda", "cwd"}
    resolved = (directory / "resolved.yaml").read_text()
    assert "# --set" in resolved and YAML(typ="safe").load(resolved) == {**DATA, "record": str(directory)}
    assert json.loads((directory / "plan.json").read_text()) == {"format": 1, "nodes": {}}
    assert sorted(path.name for path in (directory / "plugins").iterdir()) == ["kalfa_record_legos.py"]
    assert (directory / "plugins" / "kalfa_record_legos.py").read_text() == source.read_text()
    with pytest.raises(RunError):
        start_record(ctx, directory, paths=[config_file])
    again = start_record(ctx, directory, paths=[config_file], parent=tmp_path / "runs" / "old")
    assert again.read_json("manifest.json")["resumed_from"] == str(tmp_path / "runs" / "old")
    assert hidden_files(directory) == []


def test_copy_plugins_named_and_packaged(tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    single = elsewhere / "kalfa_record_single.py"
    single.write_text("VALUE = 1\n")
    load_module(single, "kalfa_record_single", monkeypatch)
    home = tmp_path / "home"
    package = home / "nested" / "kalfa_record_pack"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("from . import extra\n")
    (package / "extra.py").write_text("VALUE = 2\n")
    (package / "__pycache__").mkdir()
    (package / "__pycache__" / "junk.pyc").write_bytes(b"junk")
    load_module(package / "__init__.py", "kalfa_record_pack", monkeypatch, package=True)
    config = make_config(directories=[home], plugins=["kalfa_record_single", "kalfa_record_missing"])
    record = tmp_path / "record"
    copied = copy_plugins(config, record, imports=["kalfa_record_pack.extra.VALUE", "json.dumps"])
    assert sorted(path.relative_to(record / "plugins").as_posix() for path in copied) == [
        "kalfa_record_single.py", "nested/kalfa_record_pack"]
    assert (record / "plugins" / "nested" / "kalfa_record_pack" / "extra.py").read_text() == "VALUE = 2\n"
    assert (record / "plugins" / "nested" / "kalfa_record_pack" / "__init__.py").exists()
    assert not (record / "plugins" / "nested" / "kalfa_record_pack" / "__pycache__").exists()
    assert not (record / "plugins" / "json.py").exists()
    assert copy_plugins(make_config(), tmp_path / "empty") == []
    assert not (tmp_path / "empty" / "plugins").exists()
