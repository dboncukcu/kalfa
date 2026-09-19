import json
import os
import re
from datetime import datetime
from pathlib import Path

import pytest

from helpers import config_path
from kalfa import __version__
from kalfa.config import load_surface, parse_sets, written_config
from kalfa.record import (
    DATETIME_TOKEN,
    Record,
    read_resolved,
    record_dir,
    resolved_text,
    resume_chain,
    resume_source,
    stamp,
    write_flow,
    write_resolved,
    write_resume_note,
)
from kalfa.std.common.files import atomic, read_lines, write_text


REFERENCE = config_path("reference")


def test_manifest_is_written_once_with_the_identity(tmp_path):
    record = Record(tmp_path / "one")
    assert record.status() is None
    assert record.is_record is False
    assert record.read_json("manifest.json") is None
    note = record.manifest("run", name="one", params={"lr": 1})
    assert sorted(note) == ["kind", "name", "params", "started", "version"]
    assert note["kind"] == "run"
    assert note["version"] == __version__
    assert note["params"] == {"lr": 1}
    datetime.fromisoformat(note["started"])
    assert json.loads(record.path("manifest.json").read_text()) == note
    assert record.manifest("sweep", name="other") == note
    assert record.read_json("manifest.json")["kind"] == "run"
    assert record.is_record is True


def test_host_note_names_the_machine_process_and_directory(tmp_path):
    record = Record(tmp_path / "one")
    note = record.host()
    assert sorted(note) == ["cwd", "hostname", "pid"]
    assert note["pid"] == os.getpid()
    assert note["cwd"] == str(Path.cwd())
    assert record.read_json("host.json") == note


def test_append_writes_one_json_line_per_call(tmp_path):
    record = Record(tmp_path / "one")
    record.append("history.jsonl", {"turn": 1, "val/rmse": 0.5})
    record.append("history.jsonl", {"turn": 2, "val/rmse": 0.25, "rules": ["to_huber"]})
    text = record.path("history.jsonl").read_text()
    assert text == '{"turn": 1, "val/rmse": 0.5}\n{"turn": 2, "val/rmse": 0.25, "rules": ["to_huber"]}\n'
    assert read_lines(record.path("history.jsonl")) == [{"turn": 1, "val/rmse": 0.5},
                                                        {"turn": 2, "val/rmse": 0.25, "rules": ["to_huber"]}]


def test_read_lines_ignores_a_partial_last_line(tmp_path):
    path = tmp_path / "history.jsonl"
    path.write_text('{"turn": 1}\n{"turn": 2}\n{"tur')
    assert read_lines(path) == [{"turn": 1}, {"turn": 2}]
    assert read_lines(tmp_path / "missing.jsonl") == []


def test_write_json_and_read_json_round_trip(tmp_path):
    record = Record(tmp_path / "one")
    record.write_json("run.json", {"status": "ok", "nodes": [1, 2]})
    assert record.read_json("run.json") == {"status": "ok", "nodes": [1, 2]}
    assert record.path("run.json") == tmp_path / "one" / "run.json"
    assert record.read_json("absent.json") is None


def test_stop_request_is_a_note_with_who_and_when(tmp_path):
    record = Record(tmp_path / "one")
    record.manifest("run")
    assert record.stop_requested() is False
    assert record.stop_note() is None
    assert record.status()["stop"] is None
    record.request_stop("cli")
    assert record.stop_requested() is True
    note = record.stop_note()
    assert sorted(note) == ["at", "by"]
    assert note["by"] == "cli"
    datetime.fromisoformat(note["at"])
    assert record.status()["stop"] == note


def test_state_is_derived_from_the_files(tmp_path):
    record = Record(tmp_path / "one")
    record.manifest("run")
    assert record.state() == "pending"
    record.append("history.jsonl", {"turn": 1})
    assert record.state() == "running"
    record.write_json("run.json", {})
    assert record.state() == "finished"
    record.write_json("run.json", {"status": "ok"})
    assert record.state() == "finished"
    record.write_json("run.json", {"status": "failed"})
    assert record.state() == "failed"
    record.write_json("run.json", {"status": "error"})
    assert record.state() == "failed"


def test_steps_file_alone_means_running(tmp_path):
    record = Record(tmp_path / "one")
    record.manifest("run")
    record.append("steps.jsonl", {"step": 1})
    assert record.state() == "running"


def test_status_reports_state_last_seen_and_stop(tmp_path):
    record = Record(tmp_path / "one")
    record.manifest("run")
    status = record.status()
    assert sorted(status) == ["last_seen", "state", "stop"]
    assert status["state"] == "pending"
    assert status["stop"] is None
    assert datetime.fromisoformat(status["last_seen"]) <= datetime.now()


def test_a_resolved_yaml_alone_makes_a_record(tmp_path):
    directory = tmp_path / "old"
    directory.mkdir()
    (directory / "resolved.yaml").write_text("record: runs/old\n")
    record = Record(directory)
    assert record.is_record is True
    assert record.status()["state"] == "pending"
    assert Record(tmp_path / "empty").is_record is False


def test_stamp_is_a_second_resolution_timestamp():
    text = stamp()
    assert len(text) == 15
    assert re.fullmatch(r"\d{8}_\d{6}", text)
    datetime.strptime(text, "%Y%m%d_%H%M%S")


def test_record_dir_fills_the_datetime_token():
    assert DATETIME_TOKEN == "$datetime$"
    assert record_dir({"record": "runs/x_$datetime$"}, when="fixed") == "runs/x_fixed"
    assert record_dir({"record": "runs/$datetime$/$datetime$"}, when="T") == "runs/T/T"
    assert record_dir({"record": "runs/plain"}) == "runs/plain"
    assert re.fullmatch(r"runs/x_\d{8}_\d{6}", record_dir({"record": "runs/x_$datetime$"}))


@pytest.mark.parametrize("config", [{"record": 5}, {"record": None}, {}])
def test_record_dir_needs_a_path(config):
    with pytest.raises(ValueError) as caught:
        record_dir(config)
    assert str(caught.value) == "record must be a path"


def test_resolved_text_annotates_the_overridden_leaves(workdir):
    surface = load_surface([REFERENCE], parse_sets(["training.epochs=5"], ["seed=3"]))
    text = resolved_text(surface)
    lines = text.splitlines()
    assert lines[0] == "params:"
    assert f"  seed: 3  # --set overrides {REFERENCE}:5" in lines
    assert f"  epochs: 5  # --set overrides {REFERENCE}:166" in lines
    assert "seed: 3" in lines
    assert "alias:" not in lines
    assert "record: runs/ref_$datetime$" in lines


def test_write_resolved_and_read_resolved_round_trip(workdir):
    surface = load_surface([REFERENCE], parse_sets(["training.epochs=5"]))
    write_resolved(workdir / "rec", surface)
    assert (workdir / "rec" / "resolved.yaml").read_text() == resolved_text(surface)
    back = read_resolved(workdir / "rec")
    assert back == written_config(surface.data)
    assert "alias" not in back
    assert back["training"]["epochs"] == 5
    assert back["training"]["turn"]["uri"] == "/turn/kalfa/alternating"


def test_write_flow_writes_the_text(tmp_path):
    write_flow(tmp_path / "rec", "flow: {}\n")
    assert (tmp_path / "rec" / "flow.yaml").read_text() == "flow: {}\n"


def test_resume_source_prefers_the_last_checkpoint_over_the_final_state(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    assert resume_source(run) is None
    (run / "final").mkdir()
    (run / "final" / "state.pt").write_bytes(b"")
    assert resume_source(run) == run / "final" / "state.pt"
    (run / "checkpoints").mkdir()
    (run / "checkpoints" / "last.pt").write_bytes(b"")
    assert resume_source(run) == run / "checkpoints" / "last.pt"


def test_resume_note_and_chain_follow_the_sources(tmp_path):
    write_resume_note(tmp_path / "two", tmp_path / "one", tmp_path / "one" / "checkpoints" / "last.pt")
    assert json.loads((tmp_path / "two" / "resume.json").read_text()) == {
        "resume_from": str(tmp_path / "one"), "checkpoint": str(tmp_path / "one" / "checkpoints" / "last.pt")}
    write_resume_note(tmp_path / "three", tmp_path / "two", tmp_path / "two" / "final" / "state.pt")
    assert resume_chain(tmp_path / "three") == [str(tmp_path / "two"), str(tmp_path / "one")]
    assert resume_chain(tmp_path / "two") == [str(tmp_path / "one")]
    assert resume_chain(tmp_path / "one") == []


def test_resume_chain_stops_on_a_loop(tmp_path):
    write_resume_note(tmp_path / "a", tmp_path / "b", "x")
    write_resume_note(tmp_path / "b", tmp_path / "a", "x")
    chain = resume_chain(tmp_path / "a")
    assert len(chain) == 101
    assert chain[:2] == [str(tmp_path / "b"), str(tmp_path / "a")]


def test_writes_are_atomic(tmp_path):
    target = tmp_path / "note.txt"
    assert write_text(target, "hello") == target
    assert target.read_text() == "hello"
    assert [path.name for path in tmp_path.iterdir()] == ["note.txt"]
    with pytest.raises(RuntimeError):
        with atomic(target) as temporary:
            assert temporary.parent == tmp_path
            assert temporary.name == f".note.txt.{os.getpid()}.tmp"
            temporary.write_text("partial")
            raise RuntimeError("boom")
    assert target.read_text() == "hello"
    assert [path.name for path in tmp_path.iterdir()] == ["note.txt"]
    with atomic(target) as temporary:
        temporary.write_text("replaced")
    assert target.read_text() == "replaced"
    assert [path.name for path in tmp_path.iterdir()] == ["note.txt"]


def test_record_writes_go_through_the_atomic_helper(tmp_path):
    record = Record(tmp_path / "one")
    record.write_text("contract.yaml", "wiring: {}\n")
    record.write_json("device.json", {"type": "cpu"})
    assert record.path("contract.yaml").read_text() == "wiring: {}\n"
    assert record.read_json("device.json") == {"type": "cpu"}
    assert sorted(path.name for path in (tmp_path / "one").iterdir()) == ["contract.yaml", "device.json"]
