from datetime import datetime

import pandas
import pytest
from records.fixtures import fake_history, fake_predictions, make_record, make_sweep, write_json, write_lines

import kalfa.records
from kalfa import api, summary
from kalfa.cli import main
from kalfa.core.errors import KalfaError
from kalfa.records import reader, stats, sweep
from kalfa.records.compare import aligned_series, compare as compare_records


def test_running_record_without_history(tmp_path):
    directory = make_record(tmp_path / "fresh", status="running", turns=0, predictions=False)
    found = reader.summary_of(directory)
    assert found["status"] == "stale"
    assert found["turn"] == 0
    assert found["total"] == 10
    assert found["last"] == {}
    assert found["best"] == {"series": "val/rmse", "mode": "min", "value": None, "turn": None}
    assert found["seconds"] == 0.0
    assert found["rules"] == []
    assert found["minimizes"] == {}
    assert found["finished"] is None
    assert reader.progress_of(directory)["turn"] == 0
    assert reader.series_of(directory) == {"turn": []}
    assert reader.updates_of(directory) == {"update": [], "turn": []}
    record = api.open(directory)
    assert record.state == "stale"
    assert record.series().empty
    assert record.steps().empty
    assert record.updates.empty
    assert record.predictions is None
    assert record.stats() is None
    found = summary.summarize([directory])
    assert found.by == "val/rmse" and found.mode == "min"
    assert found.table["best"].isna().all()
    assert found.best is None
    assert "no records" not in summary.text_of(found)


def test_cli_best_line_without_a_value_says_so(tmp_path, capsys):
    directory = make_record(tmp_path / "fresh", status="running", turns=0, predictions=False)
    assert main(["summarize", str(directory)]) == 0
    out = capsys.readouterr().out
    assert "None" not in out
    assert "best: none of the records has a value" in out


def test_history_with_nan_and_string_series(tmp_path):
    directory = make_record(tmp_path / "noisy", turns=3)
    lines = fake_history(3)
    lines[1]["val/rmse"] = float("nan")
    lines[1]["seconds"] = float("nan")
    lines[2]["train/mse"] = float("nan")
    for line in lines:
        line["note"] = "text"
        line["flag"] = True
    write_lines(directory / "history.jsonl", lines)
    series = reader.series_of(directory)
    assert set(series) == {"turn", "train/mse", "val/mse", "val/rmse", "test/rmse", "lr/main"}
    assert series["turn"] == [1, 2, 3]
    assert series["val/rmse"][1] is None and series["train/mse"][2] is None
    assert series["val/rmse"][0] == lines[0]["val/rmse"]
    found = reader.summary_of(directory)
    assert found["last"]["train/mse"] is None
    assert found["last"]["val/rmse"] == lines[2]["val/rmse"]
    assert found["seconds"] == 1.0
    assert found["best"]["value"] == min(lines[0]["val/rmse"], lines[2]["val/rmse"])
    assert found["best"]["turn"] == 3
    assert found["minimizes"] == {"main": "huber"}
    assert found["rules"] == ["to_huber"]
    table = summary.summarize([directory]).table
    assert table["last_train/mse"].isna().all()
    assert table["best"].tolist() == [found["best"]["value"]]
    groups = sweep.group_folds([{**found, "params": {"fold": 0}}])
    assert "train/mse" not in groups[0]["mean"] and groups[0]["std"]["val/rmse"] == 0.0
    assert api.open(directory).series("val/rmse")["val/rmse"].isna().tolist() == [False, True, False]


def test_state_of_boundaries_and_now_forms(tmp_path):
    directory = make_record(tmp_path / "live", status="running", turns=2)
    manifest = reader.read_manifest(directory)
    assert reader.state_of(manifest, now="2026-11-07T10:06:00") == "running"
    assert reader.state_of(manifest, now="2026-11-07T10:07:00") == "running"
    assert reader.state_of(manifest, now="2026-11-07T10:07:01") == "stale"
    assert reader.state_of(manifest, now=datetime(2026, 11, 7, 10, 5, 30), stale_after=10) == "stale"
    assert reader.state_of(manifest, now=datetime(2026, 11, 7, 10, 5, 30), stale_after=60) == "running"
    assert reader.state_of(manifest) == "stale"
    assert reader.summary_of(directory)["status"] == "stale"
    assert api.open(directory).state == "stale"
    assert reader.state_of({**manifest, "status": "done"}) == "done"
    assert reader.state_of({**manifest, "status": "paused"}) == "unknown"
    assert reader.state_of(None) == "unknown"
    root, _ = make_sweep(tmp_path / "sweep", count=2)
    for point in sweep.point_dirs(root):
        write_json(point / "manifest.json", {**reader.read_manifest(point), "status": "running", "finished": None})
    assert [point["status"] for point in sweep.sweep_points(root)] == ["stale", "stale"]


def test_sweep_root_whose_last_point_failed(tmp_path):
    root, _ = make_sweep(tmp_path / "sweep", count=4)
    objective = sweep.read_sweep(root)["objective"]
    points = sweep.sweep_points(root)
    assert [point["id"] for point in points] == [0, 1, 2, 3]
    assert [point["path"] for point in points] == ["0000", "0001", "0002", "0003"]
    assert points[3]["status"] == "failed" and points[3]["objective"] is None
    assert points[3]["summary"]["failure"] == "boom"
    assert points[3]["summary"]["best"]["value"] is not None
    assert all(isinstance(point["objective"], float) for point in points[:3])
    ranked = sweep.rank_points(points, objective)
    assert ranked[-1]["id"] == 3
    assert [point["objective"] for point in ranked[:3]] == sorted(point["objective"] for point in points[:3])
    assert sweep.rank_points(points, "min") == ranked
    best = sweep.best_point(points, objective)
    assert best["id"] == ranked[0]["id"] != 3
    found = summary.summarize([root])
    assert found.sweep["points"] == 4 and len(found.points) == 4
    assert found.table["status"].tolist().count("failed") == 1
    assert found.table.set_index("name").loc["0003", "status"] == "failed"


def test_sweep_objective_watch_apart_from_the_checkpoint_watch(tmp_path):
    objective = {"watch": "val/mse", "mode": "min", "at": "best"}
    root, directories = make_sweep(tmp_path / "sweep", count=2, objective=objective)
    points = sweep.sweep_points(root)
    assert [point["summary"]["best"]["series"] for point in points] == ["val/rmse", "val/rmse"]
    expected = [reader.best_in(reader.read_history(directory), "val/mse", "min")[0] for directory in directories]
    assert [point["objective"] for point in points] == expected
    assert sweep.best_point(points, objective)["objective"] == min(expected)
    assert [point["id"] for point in sweep.rank_points(points, objective)] == sorted(range(2), key=expected.__getitem__)
    at_last = {**objective, "at": "last"}
    write_json(root / "sweep.json", {**sweep.read_sweep(root), "objective": at_last})
    points = sweep.sweep_points(root)
    assert [point["objective"] for point in points] == [point["summary"]["last"]["val/mse"] for point in points]
    assert sweep.best_point(points, at_last)["objective"] == min(point["objective"] for point in points)


def test_group_folds_single_member_and_missing_series(tmp_path):
    one = make_record(tmp_path / "one", params={"fold": 0, "lr": 0.1})
    only = reader.summary_of(one)
    groups = sweep.group_folds([only])
    assert len(groups) == 1
    assert groups[0]["group"] == {"lr": 0.1}
    assert groups[0]["members"] == [only]
    assert groups[0]["mean"] == dict(only["last"])
    assert set(groups[0]["std"]) == set(only["last"])
    assert all(value == 0.0 for value in groups[0]["std"].values())
    two = make_record(tmp_path / "two", params={"fold": 1, "lr": 0.1}, seed=1)
    other = reader.summary_of(two)
    other["last"].pop("test/rmse")
    groups = sweep.group_folds([only, other])
    assert len(groups) == 1 and len(groups[0]["members"]) == 2
    assert "test/rmse" not in groups[0]["mean"] and "test/rmse" not in groups[0]["std"]
    values = [only["last"]["val/rmse"], other["last"]["val/rmse"]]
    assert groups[0]["mean"]["val/rmse"] == pytest.approx(sum(values) / 2)
    assert groups[0]["std"]["val/rmse"] == pytest.approx(abs(values[0] - values[1]) / 2)
    plain = make_record(tmp_path / "three", params={"lr": 0.1})
    assert sweep.group_folds([reader.summary_of(plain)]) == []
    assert summary.summarize([one, plain]).folds == sweep.group_folds([only])


def test_series_and_updates_limits(tmp_path):
    directory = make_record(tmp_path / "long", turns=5)
    history = reader.read_history(directory)
    assert reader.series_of(directory, limit=2)["turn"] == [1, 5]
    chosen = reader.series_of(directory, ["val/rmse"], limit=2)
    assert chosen == {"turn": [1, 5], "val/rmse": [history[0]["val/rmse"], history[4]["val/rmse"]]}
    assert reader.series_of(directory, limit=3)["turn"] == [1, 3, 5]
    assert reader.series_of(directory, limit=100)["turn"] == [1, 2, 3, 4, 5]
    assert reader.series_of(directory, limit=5)["turn"] == [1, 2, 3, 4, 5]
    assert reader.series_of(directory, limit=None)["turn"] == [1, 2, 3, 4, 5]
    assert reader.series_of(directory, limit=0)["turn"] == [1, 2, 3, 4, 5]
    updates = reader.updates_of(directory, limit=2)
    assert updates["update"] == [1, 20] and updates["turn"] == [1, 5]
    assert updates["loss/main"] == [1.0, 0.05]
    assert reader.updates_of(directory, limit=50)["update"] == list(range(1, 21))
    record = api.open(directory)
    assert len(record.series(limit=2)) == 2 and len(record.series()) == 5
    assert len(record.steps(limit=2)) == 2 and len(record.steps()) == 20
    assert list(record.series("val/rmse", limit=2).columns) == ["turn", "val/rmse"]
    assert list(record.steps(["loss/main"]).columns) == ["update", "turn", "loss/main"]


def test_events_of_unknown_kind_and_limit(tmp_path):
    directory = make_record(tmp_path / "events")
    assert reader.events_of(directory, kinds="nope") == []
    assert reader.events_of(directory, kinds=["nope"], limit=2) == []
    assert reader.events_of(directory, kinds=[]) == []
    assert [event["kind"] for event in reader.events_of(directory, kinds=["patch", "nope"])] == ["patch"]
    assert [event["kind"] for event in reader.events_of(directory, limit=2)] == ["checkpoint", "run_end"]
    ends = reader.events_of(directory, kinds=("run_start", "run_end"), limit=1)
    assert [event["kind"] for event in ends] == ["run_end"]
    assert reader.events_of(tmp_path / "missing", kinds="patch") == []
    assert [event["by"] for event in api.open(directory).patches()] == ["to_huber"]


def test_files_of_nested_directories_and_hidden_files(tmp_path):
    directory = make_record(tmp_path / "files", predictions=False)
    (directory / "nested" / "deeper").mkdir(parents=True)
    (directory / "nested" / "deeper" / "leaf.txt").write_text("leaf", encoding="utf-8")
    (directory / "nested" / ".hidden").write_text("secret", encoding="utf-8")
    (directory / ".cache").mkdir()
    (directory / ".cache" / "inside.txt").write_text("x", encoding="utf-8")
    (directory / ".DS_Store").write_bytes(b"")
    found = reader.files_of(directory)
    paths = [entry["path"] for entry in found]
    assert paths == sorted(paths) and len(paths) == len(set(paths))
    assert "nested/deeper/leaf.txt" in paths and "checkpoints/last.pt" in paths and "manifest.json" in paths
    assert not any(part.startswith(".") for path in paths for part in path.split("/"))
    leaf = next(entry for entry in found if entry["path"] == "nested/deeper/leaf.txt")
    assert leaf["size"] == 4
    assert datetime.fromisoformat(leaf["modified"]).year == datetime.now().year
    assert set(found[0]) == {"path", "size", "modified"}
    assert reader.files_of(tmp_path / "missing") == []
    assert api.open(directory).files == found


def test_prediction_stats_with_a_single_class_present(tmp_path):
    directory = make_record(tmp_path / "binary", classes=("a", "b"))
    frame = fake_predictions(rows=20, classes=("a", "b"))
    frame["label"] = "a"
    frame.to_parquet(directory / "predictions.parquet", index=False)
    found = stats.prediction_stats(directory)
    assert found["rows"] == 20
    pair = found["pairs"][0]
    assert pair["kind"] == "classification" and pair["scores"] == ["prob_net_a", "prob_net_b"]
    result = pair["stats"]
    assert result["count"] == 20
    assert result["classes"] == ["a", "b"]
    assert result["roc"] is None and result["pr"] is None
    assert sum(result["confusion"][0]) == 20 and result["confusion"][1] == [0, 0]
    assert result["per_class"]["b"]["support"] == 0 and result["per_class"]["b"]["recall"] == 0.0
    assert result["per_class"]["a"]["recall"] == pytest.approx(result["accuracy"])
    assert result["accuracy"] == pytest.approx(result["confusion"][0][0] / 20)
    frame["pred_net"] = "a"
    frame["prob_net_a"], frame["prob_net_b"] = 0.9, 0.1
    frame.to_parquet(directory / "predictions.parquet", index=False)
    result = stats.prediction_stats(directory)["pairs"][0]["stats"]
    assert result["accuracy"] == 1.0 and result["roc"] is None and result["per_class"]["a"]["f1"] == 1.0
    plan = reader.read_plan(directory)
    plan["data"]["classes"] = None
    result = stats.prediction_stats(directory, plan=plan)["pairs"][0]["stats"]
    assert result["classes"] == ["a"] and result["confusion"] == [[20]] and result["roc"] is None


def test_classification_without_prob_columns(tmp_path):
    directory = make_record(tmp_path / "three", classes=("a", "b", "c"))
    frame = fake_predictions(rows=30, classes=("a", "b", "c"))
    frame = frame[[column for column in frame.columns if not column.startswith(("prob_", "raw_"))]]
    frame.to_parquet(directory / "predictions.parquet", index=False)
    pairs = stats.prediction_pairs(frame, reader.read_plan(directory))
    assert pairs == [{"output": "net", "pred": "pred_net", "target": "label", "kind": "classification", "raw": None,
                      "scores": []}]
    result = stats.prediction_stats(directory)["pairs"][0]["stats"]
    assert result["classes"] == ["a", "b", "c"] and result["roc"] is None and result["pr"] is None
    assert sum(map(sum, result["confusion"])) == 30
    assert sum(entry["support"] for entry in result["per_class"].values()) == 30
    plan = reader.read_plan(directory)
    plan["data"]["classes"] = None
    assert stats.prediction_pairs(frame, plan)[0]["kind"] == "classification"
    binary = make_record(tmp_path / "two", classes=("a", "b"))
    frame = fake_predictions(rows=30, classes=("a", "b"))[["row", "label", "pred_net", "town"]]
    frame.to_parquet(binary / "predictions.parquet", index=False)
    result = stats.prediction_stats(binary)["pairs"][0]["stats"]
    assert result["classes"] == ["a", "b"] and result["roc"] is None and result["pr"] is None
    assert result["count"] == 30 and 0.0 <= result["accuracy"] <= 1.0


def test_prediction_stats_with_the_plan_shape_the_kernel_writes(tmp_path):
    directory = make_record(tmp_path / "real", classes=("no", "yes"))
    plan = reader.read_plan(directory)
    plan["data"] = {"inputs": ["x"], "exposes": ["classes"], "fields": {"x0": "double", "label": "string"}}
    write_json(directory / "plan.json", plan)
    found = stats.prediction_stats(directory)
    pair = found["pairs"][0]
    assert pair["kind"] == "classification" and pair["target"] == "label"
    assert pair["stats"]["classes"] == ["no", "yes"]
    assert pair["stats"]["roc"]["auc"] > 0.5 and 0.0 < pair["stats"]["pr"]["ap"] <= 1.0
    regression = make_record(tmp_path / "reg")
    plan = reader.read_plan(regression)
    plan["data"] = {"inputs": ["x"], "exposes": [], "fields": {"x0": "double", "price": "double"}}
    write_json(regression / "plan.json", plan)
    pair = stats.prediction_stats(regression)["pairs"][0]
    assert pair["kind"] == "regression" and pair["target"] == "price" and pair["raw"] == "raw_net"


def test_prediction_stats_with_an_empty_selection(tmp_path):
    regression = make_record(tmp_path / "reg")
    found = stats.prediction_stats(regression, where="row < 0")
    assert found["rows"] == 0
    result = found["pairs"][0]["stats"]
    assert result["count"] == 0 and result["rmse"] is None and result["r2"] is None
    assert result["scatter"] == {"row": [], "truth": [], "pred": []} and result["worst"] == []
    assert result["residual_quantiles"]["p50"] is None
    assert len(result["pred_hist"]["counts"]) == 40 and sum(result["pred_hist"]["counts"]) == 0
    binary = make_record(tmp_path / "bin", classes=("a", "b"))
    result = stats.prediction_stats(binary, where="row < 0")["pairs"][0]["stats"]
    assert result["count"] == 0 and result["accuracy"] is None and result["roc"] is None
    assert result["confusion"] == [[0, 0], [0, 0]]
    assert api.open(binary).stats(where="town == 'ank'")["rows"] == 25
    assert api.open(binary).stats(where="town == 'ank'")["pairs"][0]["stats"]["count"] == 25


def test_compare_records_of_different_lengths(tmp_path):
    short = make_record(tmp_path / "short", turns=2)
    long = make_record(tmp_path / "long", turns=5, params={"lr": 0.01}, seed=1)
    found = compare_records([short, long], keys=["val/rmse"])
    assert found["series"]["turn"] == [1, 2, 3, 4, 5]
    assert found["series"][str(short)]["val/rmse"][2:] == [None, None, None]
    assert None not in found["series"][str(short)]["val/rmse"][:2]
    assert None not in found["series"][str(long)]["val/rmse"]
    assert found["params"] == {"shared": {}, "differing": {"lr": [0.001, 0.01]}}
    assert list(found["diffs"]) == ["short vs long"]
    assert "-  lr: 0.001" in found["diffs"]["short vs long"] and "+  lr: 0.01" in found["diffs"]["short vs long"]
    assert [record["turn"] for record in found["records"]] == [2, 5]
    via_api = api.open(long).compare(short, keys="val/rmse")
    assert via_api["series"]["turn"] == [1, 2, 3, 4, 5]
    assert via_api["series"][str(short)]["val/rmse"][2:] == [None, None, None]
    assert list(via_api["diffs"]) == ["long vs short"]
    default = compare_records([short, long])
    assert set(default["series"][str(long)]) == set(default["series"][str(short)]) == set(short_keys(short))


def short_keys(directory):
    return [key for key in reader.series_of(directory) if key != "turn"]


def test_summarize_over_records_and_a_sweep_root(tmp_path):
    single = make_record(tmp_path / "single", params={"lr": 0.5})
    root, points = make_sweep(tmp_path / "sweep", count=3)
    found = summary.summarize([single, root])
    assert len(found.table) == 4
    assert set(found.table["name"]) == {"single", "0000", "0001", "0002"}
    assert list(found.table.columns[:5]) == ["name", "status", "turn", "total", "seconds"]
    assert {"lr", "width", "best", "best_turn", "last_val/rmse"} <= set(found.table.columns)
    values = found.table["best"].tolist()
    assert values[:3] == sorted(values[:3]) and pandas.isna(values[3])
    assert found.table["status"].tolist()[3] == "failed"
    assert found.sweep["points"] == 3 and [point["id"] for point in found.points] == [0, 1, 2]
    assert pandas.isna(found.table.set_index("name").loc["single", "width"])
    assert found.table.set_index("name").loc["single", "lr"] == 0.5
    directories, _, _ = summary.expand([single, root, points[0]])
    assert directories == [str(single), *(str(point) for point in points)]
    whole = summary.summarize([tmp_path])
    assert set(whole.table["name"]) == set(found.table["name"]) and whole.sweep is None and whole.points == []
    assert [record.directory for record in api.records(tmp_path)] == [str(single), *(str(point) for point in points)]
    assert len(api.records(root)) == 3 and api.records(tmp_path / "missing") == []


def test_cli_top_larger_than_the_rows(tmp_path, capsys):
    root, _ = make_sweep(tmp_path / "sweep", count=2)
    assert main(["summarize", str(root), "--top", "99"]) == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert lines[0].startswith("name") and len(lines) == 4
    assert lines[-1].startswith("best: 000") and "val/rmse" in lines[-1] and lines[-1].endswith("(min)")
    assert main(["summarize", str(root), "--top", "99", "--markdown"]) == 0
    marked = capsys.readouterr().out.splitlines()
    assert len([line for line in marked if line.startswith("|")]) == 4
    assert main(["summarize", str(root), "--top", "1", "--mode", "max"]) == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 3 and "(max)" in lines[-1]
    assert summary.summarize([root], top=99).table.shape[0] == 2
    assert summary.summarize([root], top=0).best is None


def test_param_named_like_a_fixed_column_keeps_the_table_columns_unique(tmp_path):
    directory = make_record(tmp_path / "exp", params={"name": "given", "status": "x", "lr": 0.1})
    table = summary.summarize([directory]).table
    assert len(set(table.columns)) == len(table.columns)
    assert table["name"].tolist() == ["exp"]
    assert table["status"].tolist() == ["done"]


def test_series_of_accepts_a_single_key_as_a_string(tmp_path):
    directory = make_record(tmp_path / "one")
    assert list(reader.series_of(directory, "val/rmse")) == ["turn", "val/rmse"]
    assert list(reader.updates_of(directory, "loss/main")) == ["update", "turn", "loss/main"]
    assert list(aligned_series([directory], "val/rmse")[str(directory)]) == ["val/rmse"]


def test_expand_rejects_a_sweep_root_without_points(tmp_path):
    root, _ = make_sweep(tmp_path / "sweep", count=0)
    with pytest.raises(KalfaError):
        summary.expand([root])
    assert main(["summarize", str(root)]) == 2


def test_records_under_orders_by_started_then_path(tmp_path):
    make_record(tmp_path / "b", started="2026-11-07T09:00:00")
    make_record(tmp_path / "a", started="2026-11-07T09:00:00")
    make_record(tmp_path / "c", started="2026-11-07T11:00:00")
    make_record(tmp_path / "nested" / "d", started="2026-11-06T09:00:00")
    make_record(tmp_path / ".hidden" / "e", started="2026-11-08T09:00:00")
    make_sweep(tmp_path / "s", count=2)
    entries = reader.records_under(tmp_path)
    assert [entry["path"] for entry in entries] == ["c", "s", "a", "b", "nested/d"]
    assert [entry["kind"] for entry in entries] == ["record", "sweep", "record", "record", "record"]
    assert entries[1]["manifest"]["points"] == 2
    assert [record.summary["name"] for record in api.records(tmp_path)] == ["c", "0000", "0001", "a", "b", "d"]
    assert [record.summary["name"] for record in api.records(tmp_path / "s")] == ["0000", "0001"]
    assert reader.records_under(tmp_path / "a")[0]["path"] == "."


def test_records_compare_module_is_reachable_as_an_attribute():
    assert callable(getattr(kalfa.records.compare, "aligned_series", None))
