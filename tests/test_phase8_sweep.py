import json
import math
from pathlib import Path

import pytest

import kalfa.board
from kalfa.board.answers import Board, jsonable
from kalfa.records import reader, sweep
from records.fixtures import make_record, make_sweep, write_json


STATIC = Path(kalfa.board.__file__).parent / "static"


def point(identity, objective, status="done", **values):
    return {"id": identity, "path": f"{identity:04d}", "values": values, "status": status, "objective": objective}


def test_the_running_best_follows_the_order_of_the_points():
    points = [point(2, 0.4), point(0, 0.5), point(1, 0.7), point(3, None, status="failed"), point(4, 0.2)]
    found = sweep.running_best(points, {"watch": "val/rmse", "mode": "min"})
    assert found == {"ids": [0, 1, 2, 3, 4], "objective": [0.5, 0.7, 0.4, None, 0.2],
                     "best": [0.5, 0.5, 0.4, 0.4, 0.2]}
    upward = sweep.running_best(points, {"watch": "val/acc", "mode": "max"})
    assert upward["best"] == [0.5, 0.7, 0.7, 0.7, 0.7]
    assert sweep.running_best([point(0, None)], None) == {"ids": [0], "objective": [None], "best": [None]}
    assert sweep.running_best([], None) == {"ids": [], "objective": [], "best": []}


def test_axes_turn_params_into_numbers_with_their_ticks():
    points = [point(0, 0.5, lr=0.01, width=32, act="relu", layers=[8, 8]),
              point(1, 0.3, lr=0.0001, width=64, act="tanh", layers=[16]),
              point(2, None, status="failed", lr=0.001, width=64, act="relu", layers=[8, 8]),
              point(3, 0.4, lr=0.001, width=32, act="relu", layers=[8, 8])]
    spec = {"objective": {"watch": "val/rmse", "mode": "min"}, "space": {"lr": {"log": [0.0001, 0.01]}}}
    found = sweep.axes_of(points, spec)
    assert found["ids"] == [0, 1, 3] and found["left_out"] == 1 and found["mode"] == "min"
    axes = {axis["key"]: axis for axis in found["axes"]}
    assert list(axes) == ["lr", "width", "act", "layers", "objective"]
    assert axes["lr"]["label"] == "lr (log)"
    assert axes["lr"]["values"] == pytest.approx([-2.0, -4.0, -3.0])
    assert axes["lr"]["ticks"] == pytest.approx([-4.0, -3.0, -2.0])
    assert axes["lr"]["text"] == ["0.0001", "0.001", "0.01"]
    assert axes["width"] == {"key": "width", "label": "width", "values": [32.0, 64.0, 32.0], "ticks": None,
                             "text": None}
    assert axes["act"]["values"] == [0.0, 1.0, 0.0] and axes["act"]["text"] == ["relu", "tanh"]
    assert axes["act"]["ticks"] == [0.0, 1.0]
    assert axes["layers"]["text"] == ["[8, 8]", "[16]"] and axes["layers"]["values"] == [0.0, 1.0, 0.0]
    assert axes["objective"]["label"] == "val/rmse" and axes["objective"]["values"] == [0.5, 0.3, 0.4]
    assert all(math.isfinite(value) for axis in found["axes"] for value in axis["values"])


def test_axes_leave_out_params_some_points_miss_and_handle_no_points():
    points = [point(0, 0.5, lr=0.01, extra=1), point(1, 0.3, lr=0.02)]
    found = sweep.axes_of(points, {"objective": {"watch": "val/acc", "mode": "max"}})
    assert [axis["key"] for axis in found["axes"]] == ["lr", "objective"] and found["mode"] == "max"
    empty = sweep.axes_of([point(0, None)], None)
    assert empty == {"ids": [], "axes": [], "left_out": 1, "mode": "min"}
    plain = sweep.axes_of([point(0, 1.0, lr=0.0)], {"space": {"lr": {"log": [0.1, 1.0]}}})
    assert plain["axes"][0]["label"] == "lr" and plain["axes"][0]["values"] == [0.0]


def test_the_sweep_answer_carries_the_order_and_the_space(tmp_path):
    root, _ = make_sweep(tmp_path / "sweep", count=4)
    answer = Board(tmp_path).sweep("sweep")
    assert answer["order"]["ids"] == [0, 1, 2, 3]
    objectives = answer["order"]["objective"]
    assert objectives[:3] == answer["objectives"][:3] and objectives[3] is None
    best = answer["order"]["best"]
    assert best[0] == objectives[0] and best[3] == min(objectives[:3]) == answer["best"]["objective"]
    assert all(later <= earlier for earlier, later in zip(best, best[1:]))
    space = answer["space"]
    assert space["ids"] == [0, 1, 2] and space["left_out"] == 1 and space["mode"] == "min"
    assert [axis["key"] for axis in space["axes"]] == ["lr", "width", "objective"]
    assert space["axes"][-1]["label"] == "val/rmse" and space["axes"][-1]["values"] == objectives[:3]
    json.dumps(jsonable(answer), allow_nan=False)


def test_sweep_points_come_from_the_cache_until_a_point_changes(tmp_path, monkeypatch):
    root, directories = make_sweep(tmp_path / "sweep", count=3)
    cache = reader.Cache()
    first = sweep.sweep_points(root, cache)
    calls = []
    summary_of = sweep.summary_of

    def counted(directory):
        calls.append(Path(directory).name)
        return summary_of(directory)

    monkeypatch.setattr(sweep, "summary_of", counted)
    assert sweep.sweep_points(root, cache) == first and calls == []
    with (directories[1] / "history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"turn": 4, "val/rmse": 0.001}) + "\n")
    again = sweep.sweep_points(root, cache)
    assert calls == ["0001"] and again[1]["objective"] == 0.001 and again[1]["turn"] == 4
    assert [item["objective"] for item in sweep.sweep_points(root)] == [item["objective"] for item in again]
    spec = json.loads((root / "sweep.json").read_text(encoding="utf-8"))
    write_json(root / "sweep.json", {**spec, "objective": {"watch": "val/mse", "mode": "min", "at": "last"}})
    calls.clear()
    other = sweep.sweep_points(root, cache)
    assert sorted(calls) == ["0000", "0001"] or sorted(calls) == ["0000", "0001", "0002"]
    assert other[0]["objective"] != again[0]["objective"]


def test_a_point_of_a_sweep_is_compared_by_its_path(tmp_path):
    make_sweep(tmp_path / "sweep", count=3)
    make_record(tmp_path / "solo")
    found = Board(tmp_path).compare(["sweep/0000", "sweep/0001", "solo"])
    assert found["names"] == ["sweep/0000", "sweep/0001", "solo"]


def test_the_sweep_page_selects_points_and_draws_the_order_and_the_space():
    page = (STATIC / "views" / "sweep.js").read_text(encoding="utf-8")
    assert 'row-key="full"' in page and "full: pointPath(props.path, point)" in page
    assert 'selectable :selected="store.selected" @update:selected="select"' in page
    assert "compareLink(chosen.value)" in page and "pickBest(count)" in page
    assert 'kind="parallel" :data="space"' in page and "orderLines(answer.value)" in page
    plot = (STATIC / "plot.js").read_text(encoding="utf-8")
    assert "export function parallel(el, data, options = {})" in plot and 'type: "parcoords"' in plot
    assert 'if (kind === "parallel") return parallel(el, data, options);' in plot
    settings = (STATIC / "components" / "settings.js").read_text(encoding="utf-8")
    assert 'kind !== "heatmap" && kind !== "parallel"' in settings
