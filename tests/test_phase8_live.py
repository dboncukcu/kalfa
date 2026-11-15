import json
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import kalfa
import kalfa.board
from helpers import housing_frame, write_parquet
from kalfa.board.answers import Board
from kalfa.commands.sweep import points as sweep_points_module
from kalfa.core import loop
from kalfa.core.record import BEAT, LOG_FILE, Pulse, RecordWriter
from kalfa.records import reader
from records.fixtures import fake_history, make_record, write_json, write_lines


STATIC = Path(kalfa.board.__file__).parent / "static"


def stamp(moment):
    return moment.isoformat(timespec="seconds")


def config_of(tmp_path, epochs=2):
    write_parquet(housing_frame(140, 0), tmp_path / "housing.parquet")
    return {
        "seed": 1, "device": "cpu",
        "data": {"use": "table", "file": str(tmp_path / "housing.parquet"), "split": [0.7, 0.15, 0.15],
                 "features": {"x*": "standard_scaler"}, "targets": {"price": "standard_scaler"}, "batch": 32},
        "models": {"net": {"use": "mlp", "hidden": [8], "out": 1}},
        "losses": {"mse": "mse"},
        "metrics": {"rmse": "rmse"},
        "optimizers": {"main": {"use": "adam", "lr": 0.001, "over": "@models.net", "minimizes": "@losses.mse"}},
        "train": {"epochs": epochs, "checkpoint": {"watch": "val/rmse"}},
        "rules": {"slower": {"when": {"use": "after_epoch", "at": 1}, "set": {"optimizers.main.lr": {"times": 0.5}}}},
        "record": str(tmp_path / "runs" / "live"),
    }


def running_record(directory, turns=3, within=None, updated=None, planned=10, started=None):
    now = datetime.now()
    make_record(directory, turns=turns, status="running", updated=stamp(updated or now), turns_planned=planned,
                started=stamp(started or now - timedelta(minutes=5)))
    if turns == 0:
        (directory / "history.jsonl").write_text("", encoding="utf-8")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if within is not None:
        manifest["within"] = within
    write_json(directory / "manifest.json", manifest)
    return directory


def test_the_writer_notes_where_a_turn_is_and_keeps_its_start(tmp_path):
    writer = RecordWriter(tmp_path / "run")
    writer.manifest(status="running", turn=0)
    assert "within" not in writer.read_json("manifest.json")
    writer.note(1, "train", 0, 40, "batches")
    first = dict(writer.within)
    assert first["turn"] == 1 and first["phase"] == "train" and first["done"] == 0 and first["total"] == 40
    assert first["unit"] == "batches" and datetime.fromisoformat(first["started"])
    writer.within["started"] = "2026-11-01T00:00:00"
    writer.note(1, "valid", 2, 5, "batches")
    assert writer.within["started"] == "2026-11-01T00:00:00" and writer.within["phase"] == "valid"
    writer.note(2, "train", 0, 40, "batches")
    assert writer.within["started"] != "2026-11-01T00:00:00"
    note = writer.manifest(turn=1)
    assert note["within"] == writer.within and writer.read_json("manifest.json")["within"]["turn"] == 2
    done = writer.manifest(status="done")
    assert "within" not in done and "within" not in writer.read_json("manifest.json")


def test_a_beat_renews_a_running_record_and_nothing_else(tmp_path):
    writer = RecordWriter(tmp_path / "run")
    assert writer.beat() is False and not (tmp_path / "run").exists()
    writer.manifest(status="running", turn=0)
    old = "2026-11-01T00:00:00"
    write_json(tmp_path / "run" / "manifest.json", {**writer.read_json("manifest.json"), "updated": old})
    assert reader.state_of(writer.read_json("manifest.json")) == "stale"
    writer.note(1, "train", 3, 9, "batches")
    assert writer.beat() is True
    note = writer.read_json("manifest.json")
    assert note["updated"] != old and reader.state_of(note) == "running"
    assert note["within"]["done"] == 3 and note["status"] == "running"
    writer.manifest(status="stopped")
    write_json(tmp_path / "run" / "manifest.json", {**writer.read_json("manifest.json"), "updated": old})
    assert writer.beat() is False and writer.read_json("manifest.json")["updated"] == old
    (tmp_path / "run" / "manifest.json").write_text("{", encoding="utf-8")
    assert writer.beat() is False


def test_the_pulse_beats_while_it_is_open_and_ends_with_it(tmp_path):
    writer = RecordWriter(tmp_path / "run")
    writer.manifest(status="running")
    beats = []
    writer.beat = lambda: beats.append(time.monotonic())
    assert BEAT == 10.0 and Pulse(writer).every == BEAT
    with Pulse(writer, every=0.01) as pulse:
        assert pulse.thread.name == "kalfa-pulse" and pulse.thread.daemon
        deadline = time.monotonic() + 5
        while len(beats) < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
    assert len(beats) >= 3 and pulse.thread is None
    assert not any(thread.name == "kalfa-pulse" for thread in threading.enumerate())
    count = len(beats)
    time.sleep(0.05)
    assert len(beats) == count


def test_a_point_in_a_long_turn_is_not_abandoned_while_it_beats(tmp_path):
    writer = RecordWriter(tmp_path / "0000")
    writer.manifest(status="running", turn=0)
    writer.write_json("stop.json", {"by": "board"})
    write_json(tmp_path / "0000" / "manifest.json",
               {**writer.read_json("manifest.json"), "updated": "2026-11-01T00:00:00"})
    assert sweep_points_module.repairable(tmp_path / "0000") is True
    assert writer.beat() is True
    assert sweep_points_module.repairable(tmp_path / "0000") is False


def test_the_log_gets_a_stamp_on_every_line(tmp_path):
    writer = RecordWriter(tmp_path / "run")
    writer.log("first")
    writer.log("two\nlines")
    lines = (tmp_path / "run" / LOG_FILE).read_text(encoding="utf-8").splitlines()
    assert [line.split(" ", 1)[1] for line in lines] == ["first", "two", "lines"]
    assert all(datetime.fromisoformat(line.split(" ", 1)[0]) for line in lines)
    assert reader.log_of(tmp_path / "run", 2) == lines[-2:] and reader.log_of(tmp_path / "missing") == []


def test_a_run_notes_its_batches_writes_a_log_and_ends_without_a_note(tmp_path, monkeypatch):
    notes = []
    turn_ends = []
    note, manifest = RecordWriter.note, RecordWriter.manifest

    def noted(self, turn, phase, done=None, total=None, unit=None):
        notes.append((turn, phase, done, total, unit))
        return note(self, turn, phase, done, total, unit)

    def written(self, **changes):
        found = manifest(self, **changes)
        if "turn" in changes and found.get("status") == "running":
            turn_ends.append(found.get("within"))
        return found

    monkeypatch.setattr(RecordWriter, "note", noted)
    monkeypatch.setattr(RecordWriter, "manifest", written)
    result = kalfa.run(config_of(tmp_path))
    record = Path(result.record)
    train = [item for item in notes if item[0] == 1 and item[1] == "train" and item[4] == "batches"]
    assert train[0][2] == 0 and train[-1][2] == train[-1][3] >= 2
    assert [item[2] for item in train] == sorted(item[2] for item in train)
    phases = [item[1] for item in notes]
    assert {"train", "valid", "test", "after", "predictions", "samples", "plots"} <= set(phases)
    assert phases.index("valid") > phases.index("train") and phases[-1] == "plots"
    assert [within["turn"] for within in turn_ends] == [1, 2]
    assert all(within["phase"] == "test" and within["done"] == within["total"] for within in turn_ends)
    final = json.loads((record / "manifest.json").read_text(encoding="utf-8"))
    assert final["status"] == "done" and "within" not in final
    lines = [line.split(" ", 1)[1] for line in (record / "log.txt").read_text(encoding="utf-8").splitlines()]
    assert lines[0].startswith("run on ") and ", pid " in lines[0] and "device" in lines[0]
    turns = [line for line in lines if line.startswith("turn ")]
    assert turns[0].startswith("turn 1/2") and "rules slower" in turns[0]
    assert turns[1].startswith("turn 2/2") and "rules" not in turns[1]
    assert lines[-1] == "done after turn 2"
    assert not any(thread.name == "kalfa-pulse" for thread in threading.enumerate())
    answer = Board(tmp_path / "runs").record("live")
    assert answer["failure"] is None and len(answer["marks"]) == 1
    assert answer["marks"][0]["turn"] == 1 and answer["marks"][0]["rules"] == ["slower"]
    assert answer["marks"][0]["update"] == train[-1][3]
    assert any(entry["path"] == "log.txt" for entry in answer["files"])
    assert answer["progress"]["within"] is None and answer["progress"]["eta"] is None


def test_the_pulse_beats_through_a_real_run_and_never_revives_it(tmp_path, monkeypatch):
    beats = []
    beat = RecordWriter.beat

    def counted(self):
        found = beat(self)
        beats.append(found)
        return found

    monkeypatch.setattr(RecordWriter, "beat", counted)
    monkeypatch.setattr("kalfa.core.record.BEAT", 0.002)
    result = kalfa.run(config_of(tmp_path, epochs=3))
    final = json.loads((Path(result.record) / "manifest.json").read_text(encoding="utf-8"))
    assert any(beats) and final["status"] == "done" and "within" not in final and final["finished"]
    assert [line["turn"] for line in reader.read_history(result.record)] == [1, 2, 3]
    assert not any(thread.name == "kalfa-pulse" for thread in threading.enumerate())


def test_updates_as_the_unit_count_updates_within_the_turn(tmp_path, monkeypatch):
    notes = []
    note = RecordWriter.note

    def noted(self, turn, phase, done=None, total=None, unit=None):
        notes.append((turn, phase, done, total, unit))
        return note(self, turn, phase, done, total, unit)

    monkeypatch.setattr(RecordWriter, "note", noted)
    config = config_of(tmp_path)
    config["train"] = {"updates": {"total": 6, "turn": 3}}
    config["rules"] = {}
    kalfa.run(config)
    train = [item for item in notes if item[1] == "train" and item[4] == "updates"]
    assert [item[2] for item in train if item[0] == 1][-1] == 3 and {item[3] for item in train} == {3}
    assert {item[0] for item in train} == {1, 2}


def test_a_failed_run_keeps_its_traceback_in_the_log_and_for_the_board(tmp_path, monkeypatch):
    def broken(ctx, set_name):
        raise RuntimeError("boom in evaluate")

    monkeypatch.setattr(loop, "evaluate", broken)
    with pytest.raises(RuntimeError, match="boom in evaluate"):
        kalfa.run(config_of(tmp_path))
    record = tmp_path / "runs" / "live"
    text = (record / "log.txt").read_text(encoding="utf-8")
    assert "failed in turn 1: RuntimeError: boom in evaluate" in text and "Traceback (most recent call last)" in text
    failure = reader.failure_of(record)
    assert failure["type"] == "RuntimeError" and failure["message"] == "boom in evaluate"
    assert "RuntimeError: boom in evaluate" in failure["traceback"]
    answer = Board(tmp_path / "runs").record("live")
    assert answer["failure"] == failure and answer["summary"]["failure"] == "boom in evaluate"
    assert "within" not in json.loads((record / "manifest.json").read_text(encoding="utf-8"))
    assert not any(thread.name == "kalfa-pulse" for thread in threading.enumerate())
    assert reader.failure_of(tmp_path / "nowhere") is None
    assert reader.failure_of(make_record(tmp_path / "old", status="failed", failure="plain text")) == {
        "type": None, "message": "plain text", "traceback": None}


def test_progress_counts_the_part_of_the_turn_in_progress(tmp_path):
    within = {"turn": 4, "phase": "train", "done": 50, "total": 200, "unit": "batches",
              "started": stamp(datetime.now())}
    record = running_record(tmp_path / "run", within=within)
    progress = reader.progress_of(record)
    assert progress["turn"] == 3 and progress["within"] == within
    assert progress["fraction"] == pytest.approx(0.325)
    assert progress["eta"] == pytest.approx(0.5 * (10 - 3.25))
    summary = reader.summary_of(record)
    assert summary["within"] == within and summary["fraction"] == pytest.approx(0.325)
    assert summary["eta"] == pytest.approx(3.375) and summary["nonfinite"] is None


def test_an_evaluation_counts_as_the_whole_turn_and_the_end_as_none(tmp_path):
    valid = running_record(tmp_path / "valid", within={"turn": 4, "phase": "valid", "done": 1, "total": 9,
                                                         "unit": "batches", "started": stamp(datetime.now())})
    progress = reader.progress_of(valid)
    assert progress["within"]["phase"] == "valid" and progress["fraction"] == pytest.approx(0.4)
    assert progress["eta"] == pytest.approx(0.5 * 6)
    after = running_record(tmp_path / "after", within={"turn": 3, "phase": "plots", "done": None, "total": None,
                                                         "unit": None, "started": stamp(datetime.now())})
    progress = reader.progress_of(after)
    assert progress["within"]["phase"] == "plots" and progress["within"]["done"] is None
    assert progress["fraction"] == pytest.approx(0.3)
    behind = running_record(tmp_path / "behind", within={"turn": 2, "phase": "train", "done": 5, "total": 10,
                                                           "unit": "batches", "started": stamp(datetime.now())})
    assert reader.progress_of(behind)["within"] is None
    unknown = running_record(tmp_path / "unknown", within={"turn": 4, "phase": "train", "done": 7, "total": None,
                                                             "unit": "batches", "started": stamp(datetime.now())})
    progress = reader.progress_of(unknown)
    assert progress["within"]["done"] == 7 and progress["fraction"] == pytest.approx(0.3)


def test_the_first_turn_estimates_the_time_left_from_its_own_pace(tmp_path):
    now = datetime.now()
    within = {"turn": 1, "phase": "train", "done": 10, "total": 40, "unit": "batches",
              "started": stamp(now - timedelta(seconds=100))}
    record = running_record(tmp_path / "run", turns=0, within=within, updated=now)
    progress = reader.progress_of(record)
    assert progress["turn"] == 0 and progress["fraction"] == pytest.approx(0.025)
    assert progress["eta"] == pytest.approx(400 * 9.75)
    fresh = running_record(tmp_path / "fresh", turns=0, updated=now,
                           within={**within, "done": 0})
    assert reader.progress_of(fresh)["eta"] is None


def test_a_finished_record_has_no_part_and_no_time_left(tmp_path):
    record = make_record(tmp_path / "done")
    manifest = json.loads((record / "manifest.json").read_text(encoding="utf-8"))
    write_json(record / "manifest.json", {**manifest, "within": {"turn": 4, "phase": "train", "done": 1,
                                                                  "total": 2}})
    progress = reader.progress_of(record)
    assert progress["within"] is None and progress["eta"] is None and progress["fraction"] == pytest.approx(0.3)


def test_the_summary_names_the_first_value_that_is_not_finite(tmp_path):
    record = make_record(tmp_path / "run", turns=4)
    history = fake_history(4)
    history[1]["train/mse"] = float("nan")
    history[1]["val/mse"] = float("inf")
    history[2]["val/rmse"] = float("-inf")
    write_lines(record / "history.jsonl", history)
    assert reader.summary_of(record)["nonfinite"] == {"turn": 2, "series": ["train/mse", "val/mse"],
                                                      "value": "nan"}
    assert reader.nonfinite_in(history[2:]) == {"turn": 3, "series": ["val/rmse"], "value": "-inf"}
    assert reader.nonfinite_in(fake_history(3)) is None
    json.dumps(kalfa.board.answers.jsonable(Board(tmp_path).record("run")), allow_nan=False)


def test_rule_turns_carry_the_turn_the_update_and_the_names():
    assert reader.rule_turns(fake_history(3)) == [{"turn": 2, "update": 8, "rules": ["to_huber"]}]
    assert reader.rule_turns([{"turn": 1, "rules": []}, {"turn": 2}]) == []


def test_last_lines_reads_a_file_from_its_end(tmp_path):
    path = tmp_path / "long.txt"
    path.write_text("".join(f"line {index}\n" for index in range(5000)), encoding="utf-8")
    assert reader.last_lines(path, 2) == b"line 4998\nline 4999\n"
    assert reader.last_lines(path, 0) == b""
    assert len(reader.last_lines(path, 10_000).splitlines()) == 5000


def test_the_stamp_of_a_record_follows_its_updates_file(tmp_path):
    record = make_record(tmp_path / "run")
    manifest = reader.read_manifest(record)
    before = reader.record_stamp(record, manifest)
    assert reader.record_stamp(record, manifest) == before
    with (record / "updates.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"update": 99, "turn": 4}) + "\n")
    assert reader.record_stamp(record, manifest) != before
    board = Board(tmp_path)
    stamps = board.stamps()
    with (record / "updates.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"update": 100, "turn": 4}) + "\n")
    assert board.stamps() != stamps


def test_the_cache_reads_a_record_again_only_when_it_changed(tmp_path, monkeypatch):
    record = make_record(tmp_path / "run")
    calls = []
    summary_of = reader.summary_of

    def counted(directory):
        calls.append(str(directory))
        return summary_of(directory)

    monkeypatch.setattr(reader, "summary_of", counted)
    cache = reader.Cache()
    first = reader.summary_with(cache, record)
    assert reader.summary_with(cache, record) is first and len(calls) == 1
    assert (cache.hits, cache.misses) == (1, 1)
    with (record / "history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"turn": 4, "val/rmse": 0.01, "seconds": 0.5}) + "\n")
    again = reader.summary_with(cache, record)
    assert again["turn"] == 4 and len(calls) == 2
    assert reader.summary_with(None, record)["turn"] == 4 and len(calls) == 3
    small = reader.Cache(limit=1)
    small.get("a", 1, lambda: "x")
    small.get("b", 1, lambda: "y")
    assert list(small.kept) == ["b"]


def test_the_board_lists_unchanged_records_from_its_cache(tmp_path, monkeypatch):
    make_record(tmp_path / "one")
    make_record(tmp_path / "two")
    board = Board(tmp_path)
    first = board.records()
    calls = []
    summary_of = reader.summary_of

    def counted(directory):
        calls.append(Path(directory).name)
        return summary_of(directory)

    monkeypatch.setattr(reader, "summary_of", counted)
    assert board.records() == first and calls == []
    with (tmp_path / "two" / "history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"turn": 4, "val/rmse": 0.01}) + "\n")
    changed = board.records()
    assert calls == ["two"]
    assert {entry["path"]: entry["summary"]["turn"] for entry in changed["records"]} == {"one": 3, "two": 4}


def test_the_client_shows_the_part_the_time_left_the_traceback_and_the_marks():
    def source(name):
        return (STATIC / name).read_text(encoding="utf-8")

    shared = source("format.js")
    for name in ("left", "withinText", "nonfiniteText", "liveNote"):
        assert f"export function {name}(" in shared
    assert "part: { type: Number, default: null }" in source("components/progress.js")
    assert ':part="progressOf(row, column).part"' in source("components/table.js")
    assert "part: isNumber(summary.fraction) ? summary.fraction : null" in source("store.js")
    record = source("views/record.js")
    assert ':part="progress.fraction"' in record and "nonfiniteText(summary.value.nonfinite)" in record
    overview = source("views/record/overview.js")
    assert "<summary>traceback</summary>" in overview and "this.record.failure" in overview
    plot = source("plot.js")
    assert "function marksOf(marks, colors)" in plot and "layout.shapes = marks.shapes" in plot
    curves = source("views/record/curves.js")
    assert "this.record.marks" in curves and "options: { xlabel, ylabel: group.title, logy, marks }" in curves
    assert 'this.marksOn(xlabel)' in curves and "toggleRules" in curves
    log = source("views/record/log.js")
    assert 'const OUTPUT = "log.txt"' in log and "this.api.fileUrl(path, OUTPUT, TAIL)" in log
    home = source("views/home.js")
    assert "liveNote(summary)" in home and "nonfiniteText(entry.summary && entry.summary.nonfinite)" in home
