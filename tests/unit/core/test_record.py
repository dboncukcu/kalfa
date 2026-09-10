"""The record protocol: whole files written atomically, lines appended one at a time, the identity and the status."""

import json

from kalfa.record import Record
from kalfa.std.common.files import append_line, atomic, read_lines, write_json


def test_whole_files_are_written_through_a_temporary_name(tmp_path):
    path = tmp_path / "deep" / "note.json"
    write_json(path, {"a": 1})
    assert json.loads(path.read_text()) == {"a": 1} and list(path.parent.iterdir()) == [path]
    try:
        with atomic(tmp_path / "broken.txt") as temporary:
            temporary.write_text("half")
            raise RuntimeError("stop")
    except RuntimeError:
        pass
    assert not (tmp_path / "broken.txt").exists() and not list(tmp_path.glob(".broken*"))


def test_lines_are_appended_and_a_partial_last_line_is_ignored(tmp_path):
    path = tmp_path / "history.jsonl"
    append_line(path, {"turn": 1})
    append_line(path, {"turn": 2})
    with path.open("a") as stream:
        stream.write('{"turn": 3, "half')
    assert [line["turn"] for line in read_lines(path)] == [1, 2]
    assert read_lines(tmp_path / "nowhere.jsonl") == []


def test_the_manifest_is_written_once_and_the_status_is_derived(tmp_path):
    record = Record(tmp_path / "run")
    assert record.status() is None and not record.is_record
    first = record.manifest("run", name="run", params={"lr": 0.1})
    second = record.manifest("point", name="other")
    assert first == second and first["kind"] == "run" and "started" in first and "version" in first
    assert record.status()["state"] == "pending"
    record.append("history.jsonl", {"turn": 1})
    assert record.status()["state"] == "running" and record.status()["last_seen"] is not None
    record.write_json("run.json", {"status": "finished"})
    assert record.status()["state"] == "finished"
    record.write_json("run.json", {"status": "failed"})
    assert record.status()["state"] == "failed"
    assert record.host()["pid"] > 0 and record.read_json("host.json")["hostname"]
