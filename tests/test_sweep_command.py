import json
import logging
import os
import shlex
import shutil
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import kalfa
from helpers import housing_frame
from kalfa.commands import sweep as sweeping
from kalfa.commands.stop import stop_record
from kalfa.core.check import check
from kalfa.core.errors import BuildError, KalfaError, ProblemsError
from kalfa.records import reader
from records.fixtures import make_record, write_lines


LADDER = """
class Ladder:
    def __init__(self, trials, start=0.01):
        self.trials = trials
        self.start = start
        self.told = []

    def ask(self, space, objective):
        trial = len(self.told)
        return trial, {"lr": self.start / 2 ** trial, "width": space["width"][trial % len(space["width"])]}

    def tell(self, trial, value):
        self.told.append((trial, value))


class Recall(Ladder):
    def resume(self, space, objective, points):
        self.resumed = list(points)
        self.told = [(trial, value) for trial, (values, value) in enumerate(self.resumed)]
        return len(self.told)
"""


def housing_config(home, epochs=1, hidden=(8,), lr=0.01):
    return {
        "seed": 3,
        "device": "cpu",
        "data": {"use": "table", "file": str(home / "housing.parquet"), "split": [0.7, 0.15, 0.15],
                 "features": {"x*": "standard_scaler"}, "targets": {"price": "standard_scaler"}, "batch": 32},
        "models": {"net": {"use": "mlp", "hidden": list(hidden), "out": 1}},
        "losses": {"mse": "mse"},
        "metrics": {"rmse": "rmse"},
        "optimizers": {"main": {"use": "adam", "lr": lr, "over": "@models.net", "minimizes": "@losses.mse"}},
        "train": {"epochs": epochs, "checkpoint": {"watch": "val/rmse"}, "report": "best"},
    }


def sweep_config(home, root, strategy="grid", space=None, epochs=1):
    config = housing_config(home, epochs=epochs)
    config["params"] = {"lr": 0.01, "width": 8}
    config["models"]["net"]["hidden"] = ["$width$"]
    config["optimizers"]["main"]["lr"] = "$lr$"
    config["sweep"] = {"strategy": strategy, "space": space or {"lr": [0.01, 0.001], "width": [4, 8]},
                       "objective": {"watch": "val/rmse", "mode": "min", "at": "best"}, "record": str(root)}
    return config


def write_config(path, config):
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def codes_of(config):
    return [problem.code for problem in check(mappings=[config]).problems]


def sweep_problems(config):
    return [problem for problem in check(mappings=[config]).problems if problem.code.startswith("sweep_")]


@pytest.fixture(scope="module")
def home(tmp_path_factory):
    root = tmp_path_factory.mktemp("sweeps")
    housing_frame(160, 0).to_parquet(root / "housing.parquet", index=False)
    return root


@pytest.fixture
def optuna_importable(monkeypatch):
    monkeypatch.setitem(sys.modules, "optuna", ModuleType("optuna"))


def fake_attempt(plan, index, values, monitor):
    directory = Path(sweeping.point_dir(plan.root, index))
    make_record(directory, params=dict(values), sweep={"root": plan.root, "id": index, "values": dict(values)},
                seed=index)
    return SimpleNamespace(record=str(directory), context=None)


def test_space_values_reads_lists_and_ranges():
    found = sweeping.space_values({"lr": {"log": [0.001, 0.1]}, "width": [4, 8], "depth": {"int": [1, 3]},
                                   "drop": {"float": [0, 0.5]}})
    assert found == {"lr": ("log", [0.001, 0.1]), "width": ("list", [4, 8]), "depth": ("int", [1, 3]),
                     "drop": ("float", [0.0, 0.5])}
    for bad in ({}, [1, 2], {"lr": []}, {"lr": 0.1}, {"lr": {"log": [0, 1]}}, {"lr": {"float": [2, 1]}},
                {"lr": {"int": [1.5, 3]}}, {"lr": {"exp": [1, 2]}}, {"lr": {"float": [1, "a"]}},
                {"lr": {"float": [1, 2], "int": [1, 2]}}):
        with pytest.raises(BuildError):
            sweeping.space_values(bad)


def test_grid_orders_points_with_the_first_key_outermost():
    points = sweeping.Grid().points({"a": [1, 2], "b": ["x", "y", "z"], "c": [True, False]})
    assert len(points) == 12
    assert points[0] == {"a": 1, "b": "x", "c": True}
    assert points[1] == {"a": 1, "b": "x", "c": False}
    assert points[2] == {"a": 1, "b": "y", "c": True}
    assert points[6] == {"a": 2, "b": "x", "c": True}
    assert points[-1] == {"a": 2, "b": "z", "c": False}
    assert all(list(point) == ["a", "b", "c"] for point in points)
    hidden = [[8], [16, 16]]
    nested = sweeping.Grid().points({"hidden": hidden})
    assert [point["hidden"] for point in nested] == hidden
    nested[0]["hidden"].append(4)
    assert hidden == [[8], [16, 16]]


def test_grid_spreads_ranges_over_steps():
    logs = sweeping.Grid(steps=3).points({"lr": {"log": [0.001, 0.1]}})
    assert [point["lr"] for point in logs] == pytest.approx([0.001, 0.01, 0.1])
    floats = sweeping.Grid(steps=5).points({"drop": {"float": [0.0, 0.4]}})
    assert [point["drop"] for point in floats] == [0.0, 0.1, 0.2, 0.3, 0.4]
    ints = sweeping.Grid(steps=5).points({"depth": {"int": [1, 3]}})
    assert [point["depth"] for point in ints] == [1, 2, 3]
    assert all(isinstance(point["depth"], int) for point in ints)
    mixed = sweeping.Grid(steps=2).points({"lr": {"log": [0.01, 0.1]}, "width": [4, 8, 16]})
    assert len(mixed) == 6 and mixed[0] == {"lr": 0.01, "width": 4} and mixed[-1] == {"lr": 0.1, "width": 16}
    with pytest.raises(BuildError, match="steps"):
        sweeping.Grid().points({"lr": {"log": [0.001, 0.1]}, "width": [4, 8]})
    for steps in (0, -1, 2.5, True):
        with pytest.raises(BuildError):
            sweeping.Grid(steps=steps)


def test_random_draws_distinct_points_within_ranges():
    space = {"lr": {"log": [0.0001, 0.1]}, "width": {"int": [2, 16]}, "act": ["relu", "tanh"],
             "drop": {"float": [0.0, 0.5]}}
    points = sweeping.Random(trials=20, seed=5).points(space)
    assert len(points) == 20
    assert len({json.dumps(point, sort_keys=True) for point in points}) == 20
    for point in points:
        assert list(point) == ["lr", "width", "act", "drop"]
        assert 0.0001 <= point["lr"] <= 0.1 and isinstance(point["lr"], float)
        assert 2 <= point["width"] <= 16 and isinstance(point["width"], int)
        assert point["act"] in ("relu", "tanh")
        assert 0.0 <= point["drop"] <= 0.5
    assert sweeping.Random(20, seed=5).points(space) == points
    assert sweeping.Random(20).points(space, seed=5) == points
    assert sweeping.Random(20, seed=6).points(space, seed=5) != points
    small = sweeping.Random(trials=5, seed=0).points({"act": ["relu", "tanh"]})
    assert sorted(point["act"] for point in small) == ["relu", "tanh"]
    ends = sweeping.Random(trials=60, seed=1).points({"depth": {"int": [1, 3]}, "act": ["a", "b", "c", "d"]})
    assert {point["depth"] for point in ends} == {1, 2, 3}
    for trials in (0, -2, 1.5):
        with pytest.raises(BuildError):
            sweeping.Random(trials=trials)


def test_plan_of_reads_the_sweep_section(home, tmp_path, optuna_importable):
    root = tmp_path / "grid"
    plan = sweeping.plan_of(mappings=[sweep_config(home, root)])
    assert plan.count == 4 and not plan.adaptive and plan.root == str(root) and plan.config_name == "sweep"
    assert plan.points == [{"lr": 0.01, "width": 4}, {"lr": 0.01, "width": 8}, {"lr": 0.001, "width": 4},
                           {"lr": 0.001, "width": 8}]
    assert plan.objective == {"watch": "val/rmse", "mode": "min", "at": "best"}
    assert isinstance(plan.strategy, sweeping.Grid) and plan.extra["seed"] == 3
    assert plan.extra["mappings"][0]["sweep"]["record"] == str(root)
    assert plan.extra["strategy"] == {"use": "grid"}
    other = sweeping.plan_of(mappings=[sweep_config(home, root)], record=str(tmp_path / "elsewhere"),
                             params=["lr=0.5"], sets=[("train.epochs", 1)])
    assert other.root == str(tmp_path / "elsewhere")
    assert other.params == (("lr", 0.5),) and other.sets == (("train.epochs", 1),)
    stepped = sweep_config(home, root, strategy={"use": "grid", "steps": 3},
                           space={"lr": {"log": [0.001, 0.1]}, "width": [4, 8]})
    path = write_config(tmp_path / "search.yaml", stepped)
    found = sweeping.plan_of(paths=[path])
    assert found.count == 6 and found.config_name == "search" and found.paths == (str(path),)
    assert [point["lr"] for point in found.points[::2]] == pytest.approx([0.001, 0.01, 0.1])
    space = {"lr": {"log": [0.0001, 0.1]}, "width": {"int": [2, 16]}}
    drawn = sweeping.plan_of(mappings=[sweep_config(home, root, strategy={"use": "random", "trials": 4},
                                                    space=space)])
    assert drawn.points == sweeping.Random(4).points(space, 3)
    asked = sweeping.plan_of(mappings=[sweep_config(home, root, strategy={"use": "optuna", "trials": 2},
                                                    space=space)])
    assert asked.adaptive and asked.count is None and asked.strategy.seed == 3 and asked.strategy.trials == 2


def test_plan_of_needs_a_valid_sweep_section(home, tmp_path):
    config = sweep_config(home, tmp_path / "root")
    del config["sweep"]["record"]
    with pytest.raises(ProblemsError, match="sweep_record"):
        sweeping.plan_of(mappings=[config])
    assert sweeping.plan_of(mappings=[config], record=str(tmp_path / "given")).root == str(tmp_path / "given")
    plain = housing_config(home)
    with pytest.raises(KalfaError, match="no sweep section"):
        sweeping.plan_of(mappings=[plain])
    broken = sweep_config(home, tmp_path / "root", strategy="nope")
    with pytest.raises(ProblemsError, match="sweep_strategy"):
        sweeping.plan_of(mappings=[broken])


def test_write_plan_writes_the_job_files(home, tmp_path):
    root = tmp_path / "jobs"
    config = write_config(tmp_path / "sweep.yaml", sweep_config(home, root))
    plan = sweeping.plan_of(paths=[str(config)], sets=["train.epochs=1"], params=["width=8"])
    lines = []
    written = sweeping.write_plan(plan, log=lines.append)
    assert all(isinstance(path, str) for path in written)
    assert [Path(path).name for path in written] == ["sweep.json", "points.jsonl", "sweep.sh", "sweep.sub"]
    assert len(lines) == 1 and "4 points" in lines[0]
    note = json.loads((root / "sweep.json").read_text())
    assert note["format"] == 1 and note["strategy"] == {"use": "grid"} and note["points"] == 4
    assert "trials" not in note and note["record"] == str(root)
    assert note["space"] == {"lr": [0.01, 0.001], "width": [4, 8]}
    assert note["objective"] == {"watch": "val/rmse", "mode": "min", "at": "best"}
    assert note["config_files"] == [os.path.abspath(config)] and note["started"]
    assert reader.read_lines(root / "points.jsonl") == [{"id": index, "values": values}
                                                        for index, values in enumerate(plan.points)]
    script = (root / "sweep.sh").read_text()
    assert script.startswith("#!/usr/bin/env bash\nset -uo pipefail\nstatus=0\n")
    assert "set -e" not in script and script.endswith('\nexit "$status"\n')
    commands = [line for line in script.splitlines() if line.startswith("kalfa sweep")]
    assert len(commands) == 4 and all(line.endswith(" || status=1") for line in commands)
    head = ["kalfa", "sweep", os.path.abspath(config), "--set", "train.epochs=1", "-p", "width=8", "--record",
            os.path.abspath(root)]
    assert commands[2] == shlex.join([*head, "--point", "2", "--quiet"]) + " || status=1"
    assert f"# {shlex.join([*head, '--repair', '--quiet'])}\n" in script
    assert os.access(root / "sweep.sh", os.X_OK)
    submit = (root / "sweep.sub").read_text()
    assert f"executable = {shutil.which('kalfa') or sys.executable}\n" in submit and "--point $(Process)" in submit
    assert submit.rstrip().endswith("queue 4") and "--set train.epochs=1" in submit
    assert (root / "logs").is_dir()
    note["started"] = "2026-01-01T00:00:00"
    (root / "sweep.json").write_text(json.dumps(note), encoding="utf-8")
    sweeping.write_plan(plan, log=lines.append)
    assert json.loads((root / "sweep.json").read_text())["started"] == "2026-01-01T00:00:00"


def test_write_plan_for_mapping_and_adaptive_configs(home, tmp_path, optuna_importable):
    root = tmp_path / "mapping"
    plan = sweeping.plan_of(mappings=[sweep_config(home, root)])
    sweeping.write_plan(plan, log=lambda text: None)
    script = (root / "sweep.sh").read_text()
    assert "# mapping config: run through the python api" in script and "--point 2" in script
    assert not any(line.startswith("kalfa") for line in script.splitlines())
    assert all(line.startswith("#") for line in (root / "sweep.sub").read_text().splitlines())
    assert json.loads((root / "sweep.json").read_text())["config_files"] == []
    asked_root = tmp_path / "asked"
    config = write_config(tmp_path / "asked.yaml", sweep_config(home, asked_root, space={"lr": {"log": [0.001, 0.1]}},
                                                                strategy={"use": "optuna", "trials": 3}))
    asked = sweeping.plan_of(paths=[str(config)])
    sweeping.write_plan(asked, log=lambda text: None)
    note = json.loads((asked_root / "sweep.json").read_text())
    assert note["points"] is None and note["trials"] == 3 and note["strategy"] == {"use": "optuna", "trials": 3}
    assert (asked_root / "points.jsonl").read_text() == ""
    commands = [line for line in (asked_root / "sweep.sh").read_text().splitlines() if line.startswith("kalfa")]
    assert len(commands) == 1 and "--point" not in commands[0]
    assert (asked_root / "sweep.sub").read_text().rstrip().endswith("queue 1")
    write_lines(asked_root / "points.jsonl", [{"id": 0, "values": {"lr": 0.01}}])
    sweeping.write_plan(asked, log=lambda text: None)
    assert reader.read_lines(asked_root / "points.jsonl") == [{"id": 0, "values": {"lr": 0.01}}]


def test_run_local_runs_every_point_and_repair_reruns_the_failed(home, tmp_path):
    root = tmp_path / "local"
    plan = sweeping.plan_of(mappings=[sweep_config(home, root)])
    lines = []
    results = sweeping.run_local(plan, log=lines.append)
    assert len(results) == 4 and all(result is not None for result in results)
    assert (root / "sweep.json").exists() and len(reader.read_lines(root / "points.jsonl")) == 4
    for index, result in enumerate(results):
        assert result.record == str(root / f"{index:04d}")
        manifest = reader.read_manifest(result.record)
        assert manifest["status"] == "done"
        assert manifest["sweep"] == {"root": str(root), "id": index, "values": plan.points[index]}
        assert manifest["params"]["lr"] == plan.points[index]["lr"]
        assert manifest["params"]["width"] == plan.points[index]["width"]
    started = json.loads((root / "sweep.json").read_text())["started"]
    kept = reader.read_manifest(root / "0000")
    with pytest.raises(KalfaError, match="done"):
        sweeping.run_point(plan, 0)
    broken = reader.read_manifest(root / "0001")
    broken["status"] = "failed"
    (root / "0001" / "manifest.json").write_text(json.dumps(broken), encoding="utf-8")
    shutil.rmtree(root / "0003")
    assert sweeping.failed_points(plan) == [1, 3]
    lines.clear()
    repaired = sweeping.run_local(plan, repair=True, log=lines.append)
    assert [Path(result.record).name for result in repaired] == ["0001", "0003"]
    assert any("2 of 4 points to repair" in line for line in lines)
    assert sweeping.failed_points(plan) == []
    assert all(reader.read_manifest(root / f"{index:04d}")["status"] == "done" for index in range(4))
    assert reader.read_manifest(root / "0000") == kept
    assert json.loads((root / "sweep.json").read_text())["started"] == started
    lines.clear()
    assert sweeping.run_local(plan, log=lines.append) == []
    assert len([line for line in lines if "skipped" in line]) == 4
    found = kalfa.summarize(str(root))
    assert len(found.table) == 4 and found.sweep is not None
    opened = kalfa.open(str(root / "0000")).export(which="last")
    assert Path(opened.path) == root / "0000" / "exports" / "net_last.pt"
    assert check(paths=[str(root / "0000" / "resolved.yaml")]).ok


def test_a_failing_point_gives_none_and_the_loop_goes_on(home, tmp_path, caplog):
    root = tmp_path / "failing"
    plan = sweeping.plan_of(mappings=[sweep_config(home, root, space={"lr": [0.01, 0.001]})])
    plan.points[0] = {"lr": 0.01, "ghost": 1}
    lines = []
    with caplog.at_level(logging.WARNING, logger="kalfa.commands"):
        results = sweeping.run_local(plan, log=lines.append)
    assert results[0] is None and results[1] is not None
    assert not any("failed" in line for line in lines)
    assert any(text.startswith("point 0000 failed") and "ghost" in text for text in caplog.messages)
    assert reader.read_manifest(root / "0001")["status"] == "done"
    assert sweeping.failed_points(plan) == [0]
    with pytest.raises(KalfaError, match="not in the plan"):
        sweeping.run_point(plan, 7)


def test_a_stale_point_is_repaired_once_a_stop_json_waits_unread(home, tmp_path):
    root = tmp_path / "stale"
    plan = sweeping.plan_of(mappings=[sweep_config(home, root)])
    stale = make_record(root / "0001", status="running")
    assert sweeping.failed_points(plan) == [0, 2, 3]
    with pytest.raises(KalfaError, match="heartbeat is stale") as caught:
        sweeping.run_point(plan, 1)
    assert f"kalfa stop {stale}" in str(caught.value) and "--repair" in str(caught.value)
    assert stop_record(stale, by="test") == [str(stale / "stop.json")]
    assert sweeping.failed_points(plan) == [0, 1, 2, 3]
    sweeping.clear_point(stale, 1)
    assert not stale.exists()
    alive = make_record(root / "0002", status="running", updated=datetime.now().isoformat(timespec="seconds"))
    (alive / "stop.json").write_text("{}", encoding="utf-8")
    assert sweeping.failed_points(plan) == [0, 1, 3]
    with pytest.raises(KalfaError, match="point 2 is running"):
        sweeping.run_point(plan, 2)


def test_failed_points_are_missing_failed_or_stopped_from_outside(home, tmp_path):
    root = tmp_path / "fake"
    plan = sweeping.plan_of(mappings=[sweep_config(home, root)])
    make_record(root / "0000", status="done")
    make_record(root / "0001", status="failed")
    make_record(root / "0002", status="stopped")
    assert sweeping.failed_points(plan) == [1, 2, 3]
    events = reader.read_lines(root / "0002" / "events.jsonl")
    for stopping, failed in (("early: plateau", [1, 3]), ("stop.json by cli", [1, 2, 3]), ("keyboard", [1, 2, 3])):
        write_lines(root / "0002" / "events.jsonl", [*events, {"kind": "run_end", "turn": 3, "stopping": stopping}])
        assert sweeping.failed_points(plan) == failed
    make_record(root / "0003", status="running")
    assert sweeping.failed_points(plan) == [1, 2]
    with pytest.raises(KalfaError, match="running"):
        sweeping.run_point(plan, 3)
    with pytest.raises(KalfaError, match="done"):
        sweeping.run_point(plan, 0)


def test_an_adaptive_strategy_asks_runs_and_tells(home, tmp_path):
    (tmp_path / "kalfa_sweep_ladder.py").write_text(LADDER, encoding="utf-8")
    root = tmp_path / "adaptive"
    config = write_config(tmp_path / "adaptive.yaml",
                          sweep_config(home, root, strategy={"use": "kalfa_sweep_ladder.Ladder", "trials": 2}))
    plan = sweeping.plan_of(paths=[str(config)])
    assert plan.adaptive and plan.count is None and type(plan.strategy).__name__ == "Ladder"
    lines = []
    results = sweeping.run_local(plan, log=lines.append)
    assert len(results) == 2 and all(result is not None for result in results)
    assert reader.read_lines(root / "points.jsonl") == [{"id": 0, "values": {"lr": 0.01, "width": 4}},
                                                        {"id": 1, "values": {"lr": 0.005, "width": 8}}]
    points = kalfa.records.sweep_points(root)
    assert [point["id"] for point in points] == [0, 1]
    assert plan.strategy.told == [(0, points[0]["objective"]), (1, points[1]["objective"])]
    assert all(value is not None for _, value in plan.strategy.told)
    assert reader.read_manifest(root / "0001")["sweep"]["values"] == {"lr": 0.005, "width": 8}
    assert sweeping.run_local(plan, log=lines.append) == [] and "trials have run" in lines[-1]
    with pytest.raises(KalfaError, match="adaptive"):
        sweeping.run_point(plan, 5)
    assert sweeping.failed_points(plan) == []


def test_an_adaptive_sweep_that_goes_on_tells_the_strategy_the_points_already_run(home, tmp_path, monkeypatch):
    (tmp_path / "kalfa_sweep_ladder.py").write_text(LADDER, encoding="utf-8")
    monkeypatch.setattr(sweeping, "attempt", fake_attempt)
    root = tmp_path / "recall"
    config = write_config(tmp_path / "recall.yaml",
                          sweep_config(home, root, strategy={"use": "kalfa_sweep_ladder.Recall", "trials": 2}))
    assert len(sweeping.run_local(sweeping.plan_of(paths=[str(config)]), log=lambda text: None)) == 2
    make_record(root / "0001", status="failed", sweep={"root": str(root), "id": 1, "values": {"lr": 0.005, "width": 8}})
    plan = sweeping.plan_of(paths=[str(config)])
    plan.strategy.trials = 3
    lines = []
    assert len(sweeping.run_local(plan, log=lines.append)) == 1
    objective = kalfa.records.sweep_points(root)[0]["objective"]
    assert objective is not None
    assert plan.strategy.resumed == [({"lr": 0.01, "width": 4}, objective), ({"lr": 0.005, "width": 8}, None)]
    assert reader.read_lines(root / "points.jsonl")[2] == {"id": 2, "values": {"lr": 0.0025, "width": 4}}
    assert any("2 points exist and the strategy was told 2 of them" in line for line in lines)
    plain = write_config(tmp_path / "plain.yaml",
                         sweep_config(home, root, strategy={"use": "kalfa_sweep_ladder.Ladder", "trials": 4}))
    lines.clear()
    assert len(sweeping.run_local(sweeping.plan_of(paths=[str(plain)]), log=lines.append)) == 1
    assert any("does not know the points already run" in line for line in lines)


def test_check_reports_every_sweep_code(home, tmp_path, optuna_importable):
    good = sweep_config(home, tmp_path / "good")
    assert check(mappings=[good]).ok and not sweep_problems(good)
    cases = [
        ("sweep_key", lambda config: config["sweep"].update(extra=1)),
        ("sweep_key", lambda config: config.update(sweep=[1, 2])),
        ("sweep_space", lambda config: config["sweep"].update(space={"lr": 0.1})),
        ("sweep_space", lambda config: config["sweep"].update(space={"lr": {"log": [0.001, 0.1]}})),
        ("sweep_param", lambda config: config["sweep"]["space"].update(depth=[1, 2])),
        ("sweep_strategy", lambda config: config["sweep"].update(strategy="nope")),
        ("sweep_strategy", lambda config: config["sweep"].update(strategy={"use": "random"})),
        ("sweep_strategy", lambda config: config["sweep"].update(strategy="mlp")),
        ("sweep_strategy", lambda config: config["sweep"].update(strategy={"use": "optuna", "trials": 2,
                                                                           "sampler": "grid"})),
        ("sweep_objective", lambda config: config["sweep"].update(objective="val/rmse")),
        ("sweep_objective", lambda config: config["sweep"].update(objective={"mode": "min"})),
        ("sweep_objective", lambda config: config["sweep"]["objective"].update(mode="sideways")),
        ("sweep_objective", lambda config: config["sweep"]["objective"].update(at="middle")),
        ("sweep_objective", lambda config: config["sweep"]["objective"].update(watch="val/nothing")),
        ("sweep_objective", lambda config: config["sweep"]["objective"].update(watch="rmse")),
        ("sweep_objective", lambda config: config["sweep"]["objective"].update(when=1)),
        ("sweep_record", lambda config: config["sweep"].pop("record")),
    ]
    for code, change in cases:
        config = sweep_config(home, tmp_path / "bad")
        change(config)
        found = sweep_problems(config)
        assert code in [problem.code for problem in found], (code, found)
        assert all(problem.path.startswith("sweep") for problem in found)
    hint = sweep_config(home, tmp_path / "bad", strategy="gird")
    assert "did you mean grid" in str(sweep_problems(hint)[0])
    asked = sweep_config(home, tmp_path / "asked", strategy={"use": "optuna", "trials": 2, "seed": 1},
                         space={"lr": {"log": [0.001, 0.1]}, "width": {"int": [2, 8]}})
    assert check(mappings=[asked]).ok
    tracked = sweep_config(home, tmp_path / "tracked")
    tracked["sweep"]["objective"] = {"watch": "train/mse", "mode": "min", "at": "last"}
    assert "sweep_objective" not in codes_of(tracked)


def test_optuna_without_the_package_says_how_to_add_it(home, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "optuna", None)
    with pytest.raises(BuildError, match=r"optuna needs the optuna package \(uv sync installs the dependencies of "
                                         r"kalfa\)"):
        sweeping.Optuna(trials=2)
    asked = sweep_config(home, tmp_path / "asked", strategy={"use": "optuna", "trials": 2})
    assert [problem.code for problem in sweep_problems(asked)] == ["sweep_strategy"]
    assert "uv sync" in str(sweep_problems(asked)[0])
    with pytest.raises(ProblemsError, match="sweep_strategy"):
        sweeping.plan_of(mappings=[asked])
    assert not (tmp_path / "asked").exists()
    with pytest.raises(BuildError):
        sweeping.Optuna(trials=2, sampler="grid")
    with pytest.raises(BuildError):
        sweeping.Optuna(trials=0)


def test_optuna_asks_within_the_space_and_tells_values():
    strategy = sweeping.Optuna(trials=3, sampler="random", seed=0)
    space = {"lr": {"log": [0.0001, 0.1]}, "width": {"int": [2, 16]}, "act": ["relu", "tanh"],
             "hidden": [[8], [16, 16]], "drop": {"float": [0.0, 0.5]}}
    for value in (0.5, None, 0.25):
        trial, values = strategy.ask(space, {"watch": "val/rmse", "mode": "max"})
        assert 0.0001 <= values["lr"] <= 0.1 and 2 <= values["width"] <= 16 and isinstance(values["width"], int)
        assert values["act"] in ("relu", "tanh") and values["hidden"] in ([8], [16, 16])
        assert 0.0 <= values["drop"] <= 0.5
        strategy.tell(trial, value)
    assert [trial.state.name for trial in strategy.study.trials] == ["COMPLETE", "FAIL", "COMPLETE"]
    assert strategy.study.direction.name == "MAXIMIZE"


def test_optuna_resumes_from_the_points_already_run():
    space = {"lr": {"log": [0.0001, 0.1]}, "width": {"int": [2, 16]}, "act": ["relu", "tanh"],
             "hidden": [[8], [16, 16]]}
    objective = {"watch": "val/rmse", "mode": "min"}
    fresh = sweeping.Optuna(trials=4, sampler="random", seed=0)
    asked = [fresh.ask(space, objective)[1] for _ in range(2)]
    strategy = sweeping.Optuna(trials=4, sampler="random", seed=0)
    outside = {"lr": 1.0, "width": 4, "act": "relu", "hidden": [8]}
    assert strategy.resume(space, objective, [(asked[0], 0.5), (asked[1], None), (outside, 0.1)]) == 2
    trials = strategy.study.trials
    assert [trial.state.name for trial in trials] == ["COMPLETE", "FAIL"] and trials[0].value == 0.5
    assert {key: trials[0].params[key] for key in ("lr", "width", "act")} == {key: asked[0][key]
                                                                              for key in ("lr", "width", "act")}
    following = strategy.ask(space, objective)[1]
    assert following != asked[0] and following != asked[1]
    assert len(strategy.study.trials) == 3


@pytest.mark.subprocess
def test_run_local_with_jobs_runs_points_in_child_processes(home, tmp_path):
    root = tmp_path / "parallel"
    config = write_config(tmp_path / "parallel.yaml", sweep_config(home, root, space={"lr": [0.01, 0.001]}))
    plan = sweeping.plan_of(paths=[str(config)])
    lines = []
    results = sweeping.run_local(plan, jobs=2, log=lines.append)
    assert len(results) == 2 and all(result is not None for result in results), lines
    for index, result in enumerate(results):
        assert result.record == str(root / f"{index:04d}") and result.history
        manifest = reader.read_manifest(result.record)
        assert manifest["status"] == "done"
        assert manifest["sweep"] == {"root": str(root), "id": index, "values": plan.points[index]}
        assert (root / "logs" / f"{index:04d}.log").exists()
    mapping = sweeping.plan_of(mappings=[sweep_config(home, tmp_path / "mapped")])
    with pytest.raises(KalfaError, match="mapping"):
        sweeping.run_local(mapping, jobs=2, log=lines.append)
    assert not (tmp_path / "mapped").exists()
