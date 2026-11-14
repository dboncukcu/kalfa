import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from records.fixtures import fake_history, make_record as make_full_record, make_sweep, write_lines as write_rows

from kalfa.records import reader


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def write_lines(path, lines, tail=""):
    path.write_text("".join(json.dumps(line) + "\n" for line in lines) + tail, encoding="utf-8")


def make_record(directory, started="2026-11-07T10:00:00", status="running", values=(0.9, 0.5, 0.7),
                checkpoint=None, plan=True, tail=""):
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "manifest.json", {
        "format": 1, "kalfa": "3.0.0.dev0", "name": directory.name, "started": started,
        "updated": "2026-11-07T10:05:00", "finished": None, "status": status, "unit": "epochs",
        "turns_planned": 10, "params": {"lr": 0.001}, "sweep": None})
    if plan:
        train = {"epochs": 10, "report": "best"}
        if checkpoint is not None:
            train["checkpoint"] = checkpoint
        write_json(directory / "plan.json", {
            "format": 1, "nodes": {"data": {"slot": "data", "use": "kalfa.std.data.table.Table",
                                            "params": {"targets": {"price": "standard_scaler"}}}},
            "train": train})
    history = [{"turn": turn, "update": turn * 4, "train/mse": 1.0 / turn, "val/rmse": value, "lr/main": 0.001,
                "minimizes/main": "losses.mse", "seconds": 0.5, "rules": []}
               for turn, value in enumerate(values, start=1)]
    write_lines(directory / "history.jsonl", history, tail)
    write_lines(directory / "events.jsonl", [{"at": started, "turn": None, "kind": "run_start"},
                                             {"at": started, "turn": 1, "kind": "turn"}])
    return directory


def test_is_record_and_json_readers(tmp_path):
    assert not reader.is_record(tmp_path)
    assert reader.read_manifest(tmp_path) is None
    assert reader.read_plan(tmp_path) is None
    record = make_record(tmp_path / "run", checkpoint={"watch": "val/rmse"})
    assert reader.is_record(record)
    assert reader.is_record(str(record))
    assert reader.read_manifest(record)["status"] == "running"
    assert reader.read_plan(record)["train"]["checkpoint"] == {"watch": "val/rmse"}
    assert reader.read_plan(str(record))["format"] == 1


def test_history_and_events_tolerate_a_truncated_last_line(tmp_path):
    assert reader.read_history(tmp_path) == []
    assert reader.read_events(tmp_path) == []
    record = make_record(tmp_path / "run", tail='{"turn": 4, "val/rm')
    history = reader.read_history(record)
    assert [line["turn"] for line in history] == [1, 2, 3]
    assert all(isinstance(line, dict) for line in history)
    assert [event["kind"] for event in reader.read_events(record)] == ["run_start", "turn"]


def test_blank_and_broken_middle_lines_are_skipped(tmp_path):
    (tmp_path / "history.jsonl").write_text('{"turn": 1}\n\n{"turn": 2, "x": \n[1, 2]\n{"turn": 3}\n',
                                            encoding="utf-8")
    assert [line["turn"] for line in reader.read_history(tmp_path)] == [1, 3]


def test_records_under_nested_roots(tmp_path):
    make_record(tmp_path / "a" / "early", started="2026-11-01T00:00:00")
    make_record(tmp_path / "b" / "c" / "late", started="2026-11-03T00:00:00")
    make_record(tmp_path / "b" / "c" / "late" / "child", started="2026-11-04T00:00:00")
    make_record(tmp_path / ".hidden" / "ghost", started="2026-11-05T00:00:00")
    make_record(tmp_path / "middle", started="2026-11-02T00:00:00")
    (tmp_path / "empty" / "deeper").mkdir(parents=True)
    (tmp_path / "a" / "note.txt").write_text("x", encoding="utf-8")
    found = reader.records_under(tmp_path)
    assert [entry["path"] for entry in found] == ["b/c/late", "middle", "a/early"]
    assert all(isinstance(entry["manifest"], dict) for entry in found)
    assert found[0]["manifest"]["name"] == "late"
    assert set(found[0]) == {"path", "manifest", "kind"}
    assert {entry["kind"] for entry in found} == {"record"}


def test_records_under_root_is_a_record_or_missing(tmp_path):
    record = make_record(tmp_path / "run")
    assert [entry["path"] for entry in reader.records_under(record)] == ["."]
    assert reader.records_under(tmp_path / "nowhere") == []


def test_records_under_tolerates_a_broken_manifest(tmp_path):
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "manifest.json").write_text("{not json", encoding="utf-8")
    make_record(tmp_path / "fine")
    found = reader.records_under(tmp_path)
    assert [entry["path"] for entry in found] == ["fine", "broken"]
    assert found[1]["manifest"] == {}


def test_progress_of(tmp_path):
    progress = reader.progress_of(make_record(tmp_path / "run"))
    assert progress["turn"] == 3
    assert progress["total"] == 10
    assert progress["status"] == "stale"
    assert progress["manifest_status"] == "running"
    assert progress["updated"] == "2026-11-07T10:05:00"
    assert progress["seconds"] == pytest.approx(1.5)
    assert progress["last"] == {"train/mse": pytest.approx(1 / 3), "val/rmse": 0.7, "lr/main": 0.001}
    assert progress["unit"] == "epochs"


def test_progress_of_a_fresh_record(tmp_path):
    progress = reader.progress_of(make_record(tmp_path / "run", values=()))
    assert progress == {"turn": 0, "total": 10, "status": "stale", "manifest_status": "running",
                        "updated": "2026-11-07T10:05:00", "seconds": 0.0, "last": {}, "unit": "epochs",
                        "within": None, "fraction": 0.0, "eta": None}


def test_progress_of_without_a_manifest(tmp_path):
    write_lines(tmp_path / "history.jsonl", [{"turn": 2, "seconds": 1.0}, {"seconds": "slow"}])
    progress = reader.progress_of(tmp_path)
    assert progress["turn"] == 2
    assert progress["total"] is None
    assert progress["status"] == "unknown"
    assert progress["manifest_status"] is None
    assert progress["unit"] is None
    assert progress["seconds"] == 1.0


def test_progress_of_agrees_with_summary_of_and_state_of(tmp_path):
    for directory in (make_full_record(tmp_path / "done"),
                      make_full_record(tmp_path / "stale", status="running", updated="2026-11-07T10:05:00"),
                      make_full_record(tmp_path / "live", status="running",
                                       updated=datetime.now().isoformat(timespec="seconds")),
                      make_full_record(tmp_path / "fresh", turns=0, status="running")):
        progress = reader.progress_of(directory)
        summary = reader.summary_of(directory)
        assert progress["status"] == summary["status"] == reader.state_of(reader.read_manifest(directory))
        for key in ("turn", "total", "unit", "updated", "seconds", "last"):
            assert progress[key] == summary[key]
        assert not {"rules", "minimizes/main", "seconds", "turn", "update"} & set(progress["last"])
    assert reader.progress_of(tmp_path / "stale")["status"] == "stale"
    assert reader.progress_of(tmp_path / "live")["status"] == "running"


def test_progress_of_falls_back_to_the_manifest_turn(tmp_path):
    record = make_full_record(tmp_path / "run", turns=4)
    (record / "history.jsonl").unlink()
    assert reader.progress_of(record)["turn"] == 4
    assert reader.summary_of(record)["turn"] == 4
    assert reader.progress_of(record)["last"] == {}


def test_best_of_minimizes_by_default(tmp_path):
    record = make_record(tmp_path / "run", checkpoint={"watch": "val/rmse"})
    assert reader.best_of(record) == {"series": "val/rmse", "mode": "min", "value": 0.5, "turn": 2}


def test_best_of_maximizes_and_skips_non_finite_values(tmp_path):
    record = make_record(tmp_path / "run", checkpoint={"watch": "val/rmse", "mode": "max"},
                         values=(0.7, float("nan"), 0.9, float("inf")))
    assert reader.best_of(record) == {"series": "val/rmse", "mode": "max", "value": 0.9, "turn": 3}
    record = make_record(tmp_path / "low", checkpoint={"watch": "val/rmse"}, values=(0.9, float("-inf"), 0.7))
    assert reader.best_of(record) == {"series": "val/rmse", "mode": "min", "value": 0.7, "turn": 3}


def test_best_of_without_a_checkpoint_or_a_plan(tmp_path):
    assert reader.best_of(make_record(tmp_path / "plain")) is None
    assert reader.best_of(make_record(tmp_path / "noplan", plan=False)) is None
    assert reader.best_of(tmp_path / "nowhere") is None


def test_best_of_before_any_value(tmp_path):
    record = make_record(tmp_path / "run", checkpoint={"watch": "val/accuracy", "mode": "max"})
    assert reader.best_of(record) == {"series": "val/accuracy", "mode": "max", "value": None, "turn": None}


def test_best_from_takes_another_series_and_mode(tmp_path):
    record = make_full_record(tmp_path / "run", turns=3)
    plan = reader.read_plan(record)
    history = reader.read_history(record)
    assert reader.best_from(plan, history) == reader.best_of(record)
    by_mse = reader.best_from(plan, history, series="val/mse", mode="max")
    assert by_mse["series"] == "val/mse" and by_mse["mode"] == "max"
    assert (by_mse["value"], by_mse["turn"]) == reader.best_in(history, "val/mse", "max")
    assert reader.best_from(None, history, series="val/mse")["mode"] == "min"
    assert reader.best_from(plan, history, mode="maximize")["mode"] == "max"
    assert reader.best_from(None, history) is None
    assert reader.best_from({"train": {"checkpoint": {"watch": "val/rmse", "mode": "MAX"}}}, [])["mode"] == "max"


def test_summary_of_reads_each_record_file_once(tmp_path, monkeypatch):
    record = make_full_record(tmp_path / "run")
    counts = {}
    original_lines, original_json = reader.read_lines, reader.read_json

    def counted_lines(path):
        counts[Path(path).name] = counts.get(Path(path).name, 0) + 1
        return original_lines(path)

    def counted_json(directory, name):
        counts[name] = counts.get(name, 0) + 1
        return original_json(directory, name)

    monkeypatch.setattr(reader, "read_lines", counted_lines)
    monkeypatch.setattr(reader, "read_json", counted_json)
    reader.summary_of(record)
    assert counts == {"manifest.json": 1, "plan.json": 1, "history.jsonl": 1}


def test_best_in_keeps_the_earliest_of_a_tie():
    history = [{"turn": 1, "val/x": 0.5}, {"turn": 2, "val/x": 0.5}, {"turn": 3, "val/x": 0.6}]
    assert reader.best_in(history, "val/x") == (0.5, 1)
    assert reader.best_in(history, "val/x", "max") == (0.6, 3)
    assert reader.best_in(history, "val/none") == (None, None)
    assert reader.best_in([{"val/x": 1.0}, {"val/x": 0.5}], "val/x") == (0.5, 2)


def test_records_under_lists_a_sweep_root_as_one_entry(tmp_path):
    make_full_record(tmp_path / "plain", started="2026-11-07T12:00:00")
    make_sweep(tmp_path / "grid", count=3)
    make_full_record(tmp_path / "older", started="2026-11-01T00:00:00")
    found = reader.records_under(tmp_path)
    assert [(entry["path"], entry["kind"]) for entry in found] == [("plain", "record"), ("grid", "sweep"),
                                                                   ("older", "record")]
    assert found[1]["manifest"]["points"] == 3
    assert found[1]["manifest"]["started"] == "2026-11-07T10:00:00"
    assert not any(entry["path"].startswith("grid/") for entry in found)
    root, _ = make_sweep(tmp_path / "alone")
    assert [(entry["path"], entry["kind"]) for entry in reader.records_under(root)] == [(".", "sweep")]


def test_state_of_final_statuses_and_unknown():
    assert reader.state_of({"status": "done"}) == "done"
    assert reader.state_of({"status": "failed"}) == "failed"
    assert reader.state_of({"status": "stopped"}) == "stopped"
    assert reader.state_of({"status": "odd"}) == "unknown"
    assert reader.state_of({}) == "unknown"
    assert reader.state_of(None) == "unknown"


def test_state_of_running_and_stale():
    manifest = {"status": "running", "updated": "2026-11-07T10:05:00"}
    assert reader.state_of(manifest, now="2026-11-07T10:06:00") == "running"
    assert reader.state_of(manifest, now="2026-11-07T10:07:00") == "running"
    assert reader.state_of(manifest, now="2026-11-07T10:07:01") == "stale"
    assert reader.state_of(manifest, now=datetime(2026, 11, 7, 10, 5, 30)) == "running"
    assert reader.state_of(manifest, now=datetime(2026, 11, 7, 10, 5, 30), stale_after=10) == "stale"
    assert reader.state_of({"status": "running"}, now="2026-11-07T10:05:00") == "stale"
    assert reader.state_of({"status": "running", "updated": "yesterday"}, now="2026-11-07T10:05:00") == "stale"
    fresh = {"status": "running", "updated": datetime.now().isoformat(timespec="seconds")}
    assert reader.state_of(fresh) == "running"
    old = {"status": "running", "updated": (datetime.now() - timedelta(hours=1)).isoformat(timespec="seconds")}
    assert reader.state_of(old) == "stale"


def test_summary_of_a_done_record(tmp_path):
    record = make_full_record(tmp_path / "run", params={"lr": 0.01, "width": 32})
    summary = reader.summary_of(record)
    assert summary["path"] == str(record)
    assert summary["name"] == "run"
    assert summary["status"] == "done"
    assert summary["unit"] == "epochs"
    assert summary["turn"] == 3
    assert summary["total"] == 10
    assert summary["started"] == "2026-11-07T10:00:00"
    assert summary["updated"] == "2026-11-07T10:05:00"
    assert summary["finished"] == "2026-11-07T10:05:00"
    assert summary["seconds"] == pytest.approx(1.5)
    assert summary["params"] == {"lr": 0.01, "width": 32}
    history = fake_history(3)
    best_turn = min(range(3), key=lambda position: history[position]["val/rmse"]) + 1
    assert summary["best"] == {"series": "val/rmse", "mode": "min", "value": history[best_turn - 1]["val/rmse"],
                               "turn": best_turn}
    assert summary["best"] == reader.best_of(record)
    assert set(summary["last"]) == {"train/mse", "val/mse", "val/rmse", "test/rmse", "lr/main"}
    assert summary["last"]["val/rmse"] == history[-1]["val/rmse"]
    assert summary["minimizes"] == {"main": "huber"}
    assert summary["rules"] == ["to_huber"]
    assert summary["device"] == "cpu"
    assert summary["host"] == "box"
    assert summary["failure"] is None
    assert summary["sweep"] is None
    assert summary["resumed_from"] is None


def test_summary_of_failed_running_and_stale_records(tmp_path):
    failed = make_full_record(tmp_path / "failed", status="failed",
                              failure={"type": "RuntimeError", "message": "boom", "traceback": "..."})
    assert reader.summary_of(failed)["status"] == "failed"
    assert reader.summary_of(failed)["failure"] == "boom"
    now = datetime.now().isoformat(timespec="seconds")
    running = make_full_record(tmp_path / "running", status="running", updated=now, finished=None)
    summary = reader.summary_of(running)
    assert summary["status"] == "running"
    assert summary["finished"] is None
    stale = make_full_record(tmp_path / "stale", status="running", updated="2026-11-07T10:05:00")
    assert reader.summary_of(stale)["status"] == "stale"


def test_summary_of_a_sweep_point_and_a_fresh_record(tmp_path):
    root, points = make_sweep(tmp_path / "grid", count=2)
    summary = reader.summary_of(points[1])
    assert summary["sweep"] == {"root": str(root), "id": 1, "values": {"lr": 0.01, "width": 64}}
    assert summary["params"] == {"lr": 0.01, "width": 64}
    fresh = make_full_record(tmp_path / "fresh", turns=0, status="running", updated="2026-11-07T10:05:00")
    summary = reader.summary_of(fresh)
    assert summary["turn"] == 0
    assert summary["seconds"] == 0.0
    assert summary["last"] == {}
    assert summary["minimizes"] == {}
    assert summary["rules"] == []
    assert summary["best"] == {"series": "val/rmse", "mode": "min", "value": None, "turn": None}
    assert summary["status"] == "stale"


def test_summary_of_falls_back_to_the_manifest_turn_and_the_directory_name(tmp_path):
    record = make_full_record(tmp_path / "run", turns=4)
    (record / "history.jsonl").unlink()
    (record / "plan.json").unlink()
    summary = reader.summary_of(record)
    assert summary["turn"] == 4
    assert summary["best"] is None
    assert summary["last"] == {}
    empty = tmp_path / "empty"
    empty.mkdir()
    summary = reader.summary_of(empty)
    assert summary["name"] == "empty"
    assert summary["status"] == "unknown"
    assert summary["turn"] == 0
    assert summary["params"] == {}
    assert summary["host"] is None


def test_series_of_all_numeric_series(tmp_path):
    record = make_full_record(tmp_path / "run", turns=3)
    series = reader.series_of(record)
    assert series["turn"] == [1, 2, 3]
    assert set(series) == {"turn", "train/mse", "val/mse", "val/rmse", "test/rmse", "lr/main"}
    history = fake_history(3)
    assert series["val/rmse"] == [line["val/rmse"] for line in history]
    assert series["lr/main"] == [0.001, 0.001, 0.001]


def test_series_of_chosen_keys_and_missing_values(tmp_path):
    write_rows(tmp_path / "history.jsonl", [{"turn": 1, "val/x": 0.5, "val/y": "text"},
                                            {"turn": 2, "val/x": float("nan"), "val/y": 0.25},
                                            {"turn": 3, "val/y": True}])
    series = reader.series_of(tmp_path, keys=["val/x", "val/y", "val/none"])
    assert series == {"turn": [1, 2, 3], "val/x": [0.5, None, None], "val/y": [None, 0.25, None],
                      "val/none": [None, None, None]}
    assert reader.series_of(tmp_path / "nowhere") == {"turn": []}
    assert reader.series_of(tmp_path / "nowhere", keys=["val/x"]) == {"turn": [], "val/x": []}


def test_series_of_downsamples_keeping_first_and_last(tmp_path):
    write_rows(tmp_path / "history.jsonl", [{"turn": turn, "val/x": float(turn)} for turn in range(1, 101)])
    series = reader.series_of(tmp_path, limit=10)
    assert len(series["turn"]) == 10
    assert series["turn"][0] == 1
    assert series["turn"][-1] == 100
    assert series["turn"] == sorted(set(series["turn"]))
    assert series["val/x"] == [float(turn) for turn in series["turn"]]
    assert len(reader.series_of(tmp_path, limit=100)["turn"]) == 100
    assert len(reader.series_of(tmp_path, limit=None)["turn"]) == 100
    assert reader.series_of(tmp_path, limit=2)["turn"] == [1, 100]


def test_updates_of(tmp_path):
    record = make_full_record(tmp_path / "run", turns=2)
    updates = reader.updates_of(record)
    assert updates["update"] == list(range(1, 9))
    assert updates["turn"] == [1, 1, 1, 1, 2, 2, 2, 2]
    assert set(updates) == {"update", "turn", "loss/main", "lr/main", "grad_norm/main"}
    assert updates["loss/main"][0] == 1.0
    assert updates["grad_norm/main"] == [0.5] * 8
    chosen = reader.updates_of(record, keys=["loss/main"], limit=3)
    assert chosen["update"] == [1, 5, 8]
    assert chosen["turn"] == [1, 2, 2]
    assert set(chosen) == {"update", "turn", "loss/main"}
    assert reader.updates_of(tmp_path / "nowhere") == {"update": [], "turn": []}


def test_events_of_filters_by_kind_and_keeps_the_last_n(tmp_path):
    record = make_full_record(tmp_path / "run")
    assert [event["kind"] for event in reader.events_of(record)] == ["run_start", "patch", "checkpoint", "run_end"]
    assert [event["kind"] for event in reader.events_of(record, kinds=["patch"])] == ["patch"]
    assert [event["kind"] for event in reader.events_of(record, kinds="patch")] == ["patch"]
    assert [event["kind"] for event in reader.events_of(record, kinds=("patch", "run_end"))] == ["patch", "run_end"]
    assert [event["kind"] for event in reader.events_of(record, limit=2)] == ["checkpoint", "run_end"]
    assert [event["kind"] for event in reader.events_of(record, kinds=["patch", "run_end"], limit=1)] == ["run_end"]
    assert reader.events_of(record, kinds=["error"]) == []
    assert reader.events_of(tmp_path / "nowhere") == []
    patch = reader.events_of(record, kinds=["patch"])[0]
    assert patch["by"] == "to_huber"
    assert patch["path"] == "optimizers.main.minimizes"


def test_files_of_lists_every_file_with_posix_paths(tmp_path):
    record = make_full_record(tmp_path / "run")
    (record / ".DS_Store").write_bytes(b"x")
    (record / ".cache").mkdir()
    (record / ".cache" / "blob").write_bytes(b"x")
    (record / "plots" / ".hidden.png").write_bytes(b"x")
    files = reader.files_of(record)
    paths = [entry["path"] for entry in files]
    assert paths == sorted(paths)
    assert "checkpoints/last.pt" in paths
    assert "plots/loss_curve.png" in paths
    assert "manifest.json" in paths
    assert "predictions.parquet" in paths
    assert not any(part.startswith(".") for path in paths for part in path.split("/"))
    assert all(set(entry) == {"path", "size", "modified"} for entry in files)
    by_path = {entry["path"]: entry for entry in files}
    assert by_path["checkpoints/last.pt"]["size"] == 1
    assert by_path["plots/loss_curve.png"]["size"] == 3
    assert by_path["manifest.json"]["size"] == (record / "manifest.json").stat().st_size
    modified = datetime.fromisoformat(by_path["manifest.json"]["modified"])
    assert abs((datetime.now() - modified).total_seconds()) < 600
    assert reader.files_of(tmp_path / "nowhere") == []


def test_model_summary_of(tmp_path):
    record = make_full_record(tmp_path / "run")
    assert reader.model_summary_of(record)["net"]["parameters"] == 97
    assert reader.model_summary_of(tmp_path) is None


def test_series_and_updates_take_a_single_key_as_a_string(tmp_path):
    record = make_full_record(tmp_path / "run")
    assert list(reader.series_of(record, "val/rmse")) == ["turn", "val/rmse"]
    assert list(reader.updates_of(record, "loss/main")) == ["update", "turn", "loss/main"]
    assert reader.series_of(record, "val/rmse")["val/rmse"] == reader.series_of(record, ["val/rmse"])["val/rmse"]


def test_fields_of_reads_the_fitted_fields_or_none(tmp_path):
    record = make_full_record(tmp_path / "run")
    assert reader.fields_of(record) is None
    fields = {"features": [{"pattern": "x*", "columns": ["x0"]}], "targets": ["price"],
              "target_fields": [{"name": "price", "width": 1, "classes": None}], "classes": None}
    (record / "fitted").mkdir()
    write_json(record / "fitted" / "fields.json", fields)
    assert reader.fields_of(record) == fields
    (record / "fitted" / "fields.json").write_text("{broken", encoding="utf-8")
    assert reader.fields_of(record) is None
    (record / "fitted" / "fields.json").write_text("[1, 2]", encoding="utf-8")
    assert reader.fields_of(record) is None


def test_plots_of_lists_the_images_under_plots(tmp_path):
    record = make_full_record(tmp_path / "run")
    (record / "plots" / "pred").mkdir()
    (record / "plots" / "pred" / "scatter.svg").write_bytes(b"<svg/>")
    (record / "plots" / "notes.txt").write_text("x", encoding="utf-8")
    (record / "plots" / ".hidden.png").write_bytes(b"png")
    found = reader.plots_of(record)
    assert [entry["name"] for entry in found] == ["loss_curve", "pred/scatter"]
    assert [entry["path"] for entry in found] == ["plots/loss_curve.png", "plots/pred/scatter.svg"]
    assert all({"name", "path", "size", "modified"} <= set(entry) for entry in found)
    assert reader.plots_of(tmp_path / "nowhere") == []
