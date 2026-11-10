import math

import pytest

from kalfa.records import sweep
from kalfa.records.reader import best_of, progress_of
from records.fixtures import make_record, make_sweep


def point(identity, objective, status="done"):
    return {"id": identity, "path": f"{identity:04d}", "values": {}, "status": status, "turn": 3, "total": 10,
            "objective": objective, "summary": {}}


def summary(params, last):
    return {"path": "x", "name": "x", "status": "done", "params": dict(params), "last": dict(last), "best": None}


def test_is_sweep_and_read_sweep(tmp_path):
    root, _ = make_sweep(tmp_path / "grid")
    assert sweep.is_sweep(root)
    assert not sweep.is_sweep(tmp_path)
    assert sweep.read_sweep(root)["objective"] == {"watch": "val/rmse", "mode": "min", "at": "best"}
    assert sweep.read_sweep(tmp_path) is None


def test_point_dirs_keeps_four_digit_directories_in_id_order(tmp_path):
    root, points = make_sweep(tmp_path / "grid", count=4)
    (root / "notes").mkdir()
    (root / "12345").mkdir()
    (root / "0007.txt").write_text("x", encoding="utf-8")
    (root / "0010").mkdir()
    (root / "0009").mkdir()
    assert [path.name for path in sweep.point_dirs(root)] == ["0000", "0001", "0002", "0003", "0009", "0010"]
    assert sweep.point_dirs(root)[:4] == points


def test_point_dirs_of_missing_or_empty_root(tmp_path):
    assert sweep.point_dirs(tmp_path / "nowhere") == []
    assert sweep.point_dirs(tmp_path) == []


def test_objective_of_at_best_needs_the_matching_series():
    best = {"series": "val/rmse", "mode": "min", "value": 0.25, "turn": 3}
    record = {"best": best, "last": {"val/rmse": 0.4, "val/mse": 0.16}}
    assert sweep.objective_of(record, {"watch": "val/rmse", "mode": "min", "at": "best"}) == 0.25
    assert sweep.objective_of(record, {"watch": "val/mse", "mode": "min", "at": "best"}) is None
    assert sweep.objective_of(record, {"watch": "val/rmse", "mode": "min"}) == 0.25


def test_objective_of_at_last_reads_the_last_row():
    record = {"best": None, "last": {"val/rmse": 0.4, "val/mse": 0.16}}
    assert sweep.objective_of(record, {"watch": "val/mse", "mode": "min", "at": "last"}) == 0.16
    assert sweep.objective_of(record, {"watch": "test/rmse", "mode": "min", "at": "last"}) is None
    assert sweep.objective_of(record, {"watch": "val/mse", "mode": "min", "at": "best"}) is None


def test_objective_of_without_value():
    assert sweep.objective_of({"best": None, "last": None}, {"watch": "val/rmse", "at": "best"}) is None
    assert sweep.objective_of({"best": None, "last": {}}, {"watch": "val/rmse", "at": "last"}) is None
    assert sweep.objective_of({"best": {"series": "val/rmse", "value": None}}, {"watch": "val/rmse"}) is None
    assert sweep.objective_of({"best": {"series": "v", "value": math.nan}}, {"watch": "v"}) is None
    assert sweep.objective_of({"best": {"series": "v", "value": 1.0}}, {"watch": None}) is None
    assert sweep.objective_of({"best": {"series": "v", "value": 1.0}}, None) is None
    assert sweep.objective_of(None, {"watch": "v"}) is None


def test_sweep_points_reads_every_point(tmp_path):
    root, dirs = make_sweep(tmp_path / "grid", count=4)
    points = sweep.sweep_points(root)
    assert [item["id"] for item in points] == [0, 1, 2, 3]
    assert [item["path"] for item in points] == ["0000", "0001", "0002", "0003"]
    assert [item["values"] for item in points] == [{"lr": 0.01, "width": 32}, {"lr": 0.01, "width": 64},
                                                   {"lr": 0.001, "width": 32}, {"lr": 0.001, "width": 64}]
    assert [item["status"] for item in points] == ["done", "done", "done", "failed"]
    assert all(item["turn"] == 3 and item["total"] == 10 for item in points)
    for item in points[:3]:
        expected = best_of(root / item["path"])
        assert expected["series"] == "val/rmse"
        assert item["objective"] == expected["value"]
        assert item["summary"]["name"] == item["path"]
    assert points[3]["objective"] is None
    assert points[3]["summary"]["failure"] == "boom"


def test_sweep_points_at_last_and_without_sweep_json(tmp_path):
    root, _ = make_sweep(tmp_path / "grid", count=2, objective={"watch": "val/mse", "mode": "min", "at": "last"})
    points = sweep.sweep_points(root)
    for item in points:
        last = progress_of(root / item["path"])["last"]
        assert item["objective"] == last["val/mse"]
    (root / "sweep.json").unlink()
    points = sweep.sweep_points(root)
    assert len(points) == 2
    assert all(item["objective"] is None for item in points)


def test_sweep_points_skips_nothing_for_an_empty_or_pending_point(tmp_path):
    root, _ = make_sweep(tmp_path / "grid", count=2)
    (root / "0002").mkdir()
    make_record(root / "0003", status="running", predictions=False, turns=1,
                sweep={"root": str(root), "id": 3, "values": {"lr": 0.5, "width": 8}})
    points = sweep.sweep_points(root)
    assert [item["id"] for item in points] == [0, 1, 2, 3]
    assert points[2]["status"] == "unknown"
    assert points[2]["objective"] is None
    assert points[2]["turn"] == 0
    assert points[3]["status"] == "stale"
    assert points[3]["values"] == {"lr": 0.5, "width": 8}
    assert points[3]["turn"] == 1


def test_sweep_points_of_a_plain_directory(tmp_path):
    assert sweep.sweep_points(tmp_path) == []
    assert sweep.sweep_points(tmp_path / "nowhere") == []


def test_rank_points_min_mode_then_unvalued_by_id():
    points = [point(3, 0.5), point(1, None), point(2, 0.2), point(0, 0.9), point(5, 0.2), point(4, 0.1, "failed")]
    ranked = sweep.rank_points(points, {"watch": "val/rmse", "mode": "min"})
    assert [item["id"] for item in ranked] == [2, 5, 3, 0, 1, 4]
    assert ranked[-1]["status"] == "failed"
    assert ranked[-1]["objective"] == 0.1


def test_rank_points_max_mode():
    points = [point(3, 0.5), point(1, None), point(2, 0.2), point(0, 0.9), point(5, 0.9)]
    ranked = sweep.rank_points(points, {"watch": "val/acc", "mode": "max"})
    assert [item["id"] for item in ranked] == [0, 5, 3, 2, 1]
    assert [item["id"] for item in sweep.rank_points(points, "max")] == [0, 5, 3, 2, 1]
    assert [item["id"] for item in sweep.rank_points(points, {})] == [2, 3, 0, 5, 1]


def test_rank_points_leaves_the_input_alone():
    points = [point(1, 0.3), point(0, 0.1)]
    sweep.rank_points(points, {"mode": "min"})
    assert [item["id"] for item in points] == [1, 0]
    assert sweep.rank_points([], {"mode": "min"}) == []


def test_best_point_of_a_sweep(tmp_path):
    root, _ = make_sweep(tmp_path / "grid", count=4)
    points = sweep.sweep_points(root)
    objective = sweep.read_sweep(root)["objective"]
    best = sweep.best_point(points, objective)
    valued = [item for item in points if item["objective"] is not None]
    assert len(valued) == 3
    assert best["objective"] == min(item["objective"] for item in valued)
    assert best["status"] == "done"
    ranked = sweep.rank_points(points, objective)
    assert ranked[0] is best
    assert ranked[-1]["status"] == "failed"
    assert [item["objective"] for item in ranked[:3]] == sorted(item["objective"] for item in valued)


def test_best_point_without_values():
    assert sweep.best_point([], {"mode": "min"}) is None
    assert sweep.best_point([point(0, None), point(1, 0.4, "failed")], {"mode": "min"}) is None
    only = point(2, 0.4)
    assert sweep.best_point([point(0, None), only], {"mode": "max"}) is only


def test_group_folds_groups_by_the_other_params():
    summaries = [
        summary({"lr": 0.01, "fold": 0}, {"val/rmse": 1.0, "test/rmse": 2.0, "minimizes/main": "mse"}),
        summary({"lr": 0.1, "fold": 0}, {"val/rmse": 5.0}),
        summary({"fold": 1, "lr": 0.01}, {"val/rmse": 3.0, "val/mse": 9.0}),
        summary({"lr": 0.01}, {"val/rmse": 100.0}),
        summary({"lr": 0.01, "fold": 2}, {"val/rmse": 2.0, "val/mse": 4.0, "test/rmse": 1.0}),
    ]
    groups = sweep.group_folds(summaries)
    assert [group["group"] for group in groups] == [{"lr": 0.01}, {"lr": 0.1}]
    first, second = groups
    assert first["members"] == [summaries[0], summaries[2], summaries[4]]
    assert first["mean"] == {"val/rmse": pytest.approx(2.0)}
    assert first["std"] == {"val/rmse": pytest.approx(math.sqrt(2.0 / 3.0))}
    assert second["members"] == [summaries[1]]
    assert second["mean"] == {"val/rmse": 5.0}
    assert second["std"] == {"val/rmse": 0.0}
    assert all("fold" not in group["group"] for group in groups)


def test_group_folds_with_another_key_and_odd_params():
    summaries = [
        summary({"lr": 0.01, "layers": [32, 16], "split": 0}, {"val/rmse": 1.0}),
        summary({"lr": 0.01, "layers": [32, 16], "split": 1}, {"val/rmse": 3.0}),
        summary({"lr": 0.01, "layers": [32, 32], "split": 0}, {"val/rmse": 7.0}),
        {"params": None, "last": {"val/rmse": 1.0}},
        {"params": {"split": 3}, "last": None},
        None,
    ]
    groups = sweep.group_folds(summaries, key="split")
    assert [group["group"] for group in groups] == [{"lr": 0.01, "layers": [32, 16]}, {"lr": 0.01, "layers": [32, 32]},
                                                    {}]
    assert groups[0]["mean"] == {"val/rmse": 2.0}
    assert groups[0]["std"] == {"val/rmse": 1.0}
    assert groups[2]["members"] == [summaries[4]]
    assert groups[2]["mean"] == {} and groups[2]["std"] == {}
    assert sweep.group_folds([]) == []
    assert sweep.group_folds([summary({"lr": 0.01}, {"val/rmse": 1.0})]) == []


def test_group_folds_over_sweep_summaries(tmp_path):
    root = tmp_path / "folds"
    for index, (lr, fold) in enumerate([(0.01, 0), (0.01, 1), (0.1, 0)]):
        make_record(root / f"{index:04d}", params={"lr": lr, "fold": fold}, seed=index,
                    sweep={"root": str(root), "id": index, "values": {"lr": lr, "fold": fold}})
    summaries = [item["summary"] for item in sweep.sweep_points(root)]
    groups = sweep.group_folds(summaries)
    assert [group["group"] for group in groups] == [{"lr": 0.01}, {"lr": 0.1}]
    assert len(groups[0]["members"]) == 2
    lasts = [member["last"] for member in groups[0]["members"]]
    for series in ("train/mse", "val/mse", "val/rmse", "test/rmse", "lr/main"):
        assert groups[0]["mean"][series] == pytest.approx(sum(last[series] for last in lasts) / 2)
        assert groups[0]["std"][series] == pytest.approx(abs(lasts[0][series] - lasts[1][series]) / 2)
    assert "minimizes/main" not in groups[0]["mean"]
    assert groups[1]["std"] == {series: 0.0 for series in groups[1]["mean"]}


def test_objective_of_at_best_falls_back_to_the_history_when_the_watch_differs(tmp_path):
    record = make_record(tmp_path / "run")
    found = sweep.summary_of(record)
    history = sweep.read_history(record)
    assert found["best"]["series"] == "val/rmse"
    assert sweep.objective_of(found, {"watch": "val/mse", "mode": "min", "at": "best"}) == min(
        line["val/mse"] for line in history)
    assert sweep.objective_of(found, {"watch": "val/rmse", "mode": "max", "at": "best"}) == max(
        line["val/rmse"] for line in history)
    assert sweep.objective_of(found, {"watch": "nothing", "mode": "min", "at": "best"}) is None
    assert sweep.objective_of({**found, "path": tmp_path / "nowhere"}, {"watch": "val/mse"}) is None


def test_is_sweep_and_mode_of_share_the_reader_helpers():
    assert sweep.is_sweep is sweep.is_sweep_root
    assert sweep.mode_of({"mode": "MAX"}) == "max" and sweep.mode_of("maximize") == "max"
    assert sweep.mode_of(None) == "min" and sweep.mode_of({}) == "min"
