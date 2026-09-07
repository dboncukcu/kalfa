"""kalfa sweep: strategies deterministic by id, the plan of config 14, one point by id, the local loop in
subprocesses, collect on a half done root, an optuna loop fed back."""

import json
import shutil
from pathlib import Path

import pytest
from ruamel.yaml import YAML

import kalfa  # noqa: F401
from conftest import ROOT
from kalfa import sweep as sweeper
from kalfa.api import check
from kalfa.cli import main
from kalfa.collect import collect
from kalfa.config import parse_sets
from kalfa.std.strategy import Grid, RandomSearch, SobolSearch, grid_values, parse_space
from kalfa.synthetic import write_housing

CONFIG = str(ROOT / "configs" / "14_sweep_grid.yaml")
OPTUNA = ["sweep.strategy={uri: optuna, params: {trials: 2, seed: 1}}",
          "sweep.space.lr={low: 1.0e-4, high: 1.0e-2, log: true}"]


def flags(sets):
    return [argument for text in sets for argument in ("--set", text)]


@pytest.fixture
def housing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_housing(tmp_path / "housing.parquet")
    return tmp_path


def test_space_parsing_and_grid_points():
    space = parse_space({"lr": [1e-2, 1e-3, 1e-4], "width": [64, 128]})
    grid = Grid()
    assert grid.total(space) == 6
    assert grid.point(space, 0) == {"lr": 1e-2, "width": 64} and grid.point(space, 1) == {"lr": 1e-2, "width": 128}
    assert grid.point(space, 4) == {"lr": 1e-4, "width": 64} and grid.point(space, 5) == {"lr": 1e-4, "width": 128}
    ranged = parse_space({"lr": {"low": 1e-4, "high": 1e-2, "log": True, "steps": 3}})
    assert grid_values("lr", ranged["lr"]) == pytest.approx([1e-4, 1e-3, 1e-2])
    with pytest.raises(ValueError, match="steps"):
        Grid().total(parse_space({"lr": {"low": 0.0, "high": 1.0}}))
    with pytest.raises(ValueError):
        grid.point(space, 6)
    for bad in ({"lr": []}, {"lr": {"low": 1, "high": 0}}, {"lr": {"low": 0, "high": 1, "log": True}}, {"lr": 3}, {},
                {"lr": {"low": 0, "high": 1, "ghost": 2}}):
        with pytest.raises(ValueError):
            parse_space(bad)


def test_random_and_sobol_are_deterministic_by_id():
    space = parse_space({"lr": {"low": 1e-4, "high": 1e-2, "log": True}, "width": [32, 64, 128],
                         "depth": {"low": 1, "high": 4, "int": True}})
    for strategy in (RandomSearch(5, seed=3), SobolSearch(5, seed=3)):
        points = [strategy.point(space, index) for index in range(5)]
        again = type(strategy)(5, seed=3)
        assert [again.point(space, index) for index in (4, 2, 0)] == [points[4], points[2], points[0]]
        assert len({json.dumps(point, sort_keys=True) for point in points}) > 1
        for point in points:
            assert 1e-4 <= point["lr"] <= 1e-2 and point["width"] in (32, 64, 128)
            assert isinstance(point["depth"], int) and 1 <= point["depth"] <= 4
        with pytest.raises(ValueError):
            strategy.point(space, 5)
    assert RandomSearch(5, seed=4).point(space, 0) != RandomSearch(5, seed=3).point(space, 0)
    assert SobolSearch(5, seed=4).point(space, 1) != SobolSearch(5, seed=3).point(space, 1)


def test_point_values_read_back_as_the_same_scalars():
    for value in (1e-05, 0.001, 3, "relu", True, 12.5):
        assert YAML(typ="safe").load(sweeper.yaml_value(value)) == value


def test_plan_count_show_and_check(housing, capsys):
    plan = sweeper.plan([CONFIG], parse_sets([]))
    assert plan.total == 6 and plan.deterministic and plan.root == Path("runs/sweep_housing")
    assert plan.uri == "/strategy/kalfa/grid" and sweeper.point_of(plan, 4) == {"lr": 1e-4, "width": 64}
    assert check([CONFIG], parse_sets([])).problems == []
    kinds = [problem.kind for problem in check([CONFIG], parse_sets(["sweep.space.ghost=[1, 2]"])).problems]
    assert "unresolved_ref" in kinds
    kinds = [problem.kind for problem in check([CONFIG], parse_sets(["sweep.objective.mode=up"])).problems]
    assert "invalid_value" in kinds
    kinds = [problem.kind for problem in check([CONFIG], parse_sets(["sweep.space.lr={low: 0.0, high: 1.0}"])).problems]
    assert "sweep_space" in kinds
    assert main(["sweep", CONFIG, "--count"]) == 0
    assert capsys.readouterr().out.strip() == "6"
    assert main(["sweep", CONFIG, "--show", "4"]) == 0
    assert json.loads(capsys.readouterr().out) == {"lr": 1e-4, "width": 64}
    assert main(["sweep", CONFIG, "--show", "4", *flags(OPTUNA)]) == 1
    assert main(["sweep", CONFIG, "--id", "0", *flags(OPTUNA)]) == 1


def test_one_point_by_id(housing):
    assert main(["sweep", CONFIG, "--id", "4", "-p", "epochs=2"]) == 0
    record = Path("runs/sweep_housing/0004")
    info = json.loads((record / "sweep.json").read_text())
    assert info["id"] == 4 and info["point"] == {"lr": 1e-4, "width": 64} and info["total"] == 6
    assert info["strategy"] == "/strategy/kalfa/grid" and info["record"] == str(record)
    assert info["objective"]["monitor"] == "val/rmse" and info["objective"]["mode"] == "min"
    assert info["objective"]["at"] == "best" and info["objective"]["turn"] in (1, 2)
    resolved = (record / "resolved.yaml").read_text()
    assert "width: 64" in resolved and "lr: 0.0001" in resolved
    assert (record / "history.jsonl").exists() and not Path("runs/sweep_housing/0000").exists()


def test_local_loop_and_collect_with_a_missing_folder(housing):
    assert main(["sweep", CONFIG, "-p", "epochs=1", "--record", "runs/grid"]) == 0
    root = Path("runs/grid")
    assert sorted(path.name for path in root.iterdir()) == [f"{index:04d}" for index in range(6)]
    shutil.rmtree(root / "0002")
    (root / "0003" / "sweep.json").unlink()
    kind, text, target = collect(["runs/grid"])
    assert kind == "sweep" and target == "runs/grid"
    assert (root / "sweep.csv").exists() and (root / "sweep.md").exists()
    summary = json.loads((root / "sweep.json").read_text())
    assert [row["id"] for row in summary["points"]] == [0, 1, 4, 5]
    assert summary["skipped"] == [{"dir": "0003", "status": "unfinished"}]
    assert summary["best"]["id"] in (0, 1, 4, 5) and "best: point" in text and "skipped: 0003" in text
    assert {"lr", "width", "objective", "turn", "turns", "val/rmse"} <= set(summary["points"][0])
    assert main(["collect", "runs/grid"]) == 0
    assert main(["sweep", CONFIG, "-p", "epochs=1", "--record", "runs/grid"]) == 1
    assert (root / "0002" / "sweep.json").exists() and not (root / "0003" / "sweep.json").exists()


def test_optuna_loop_feeds_the_objective_back(housing):
    assert main(["sweep", CONFIG, "-p", "epochs=1", "--record", "runs/opt", *flags(OPTUNA)]) == 0
    root = Path("runs/opt")
    points = [json.loads((root / f"{index:04d}" / "sweep.json").read_text()) for index in range(2)]
    assert all(1e-4 <= point["point"]["lr"] <= 1e-2 for point in points)
    assert all(point["point"]["width"] in (64, 128) for point in points)
    assert all(point["strategy"] == "/strategy/kalfa/optuna" and point["total"] == 2 for point in points)
    kind, text, target = collect(["runs/opt"])
    assert kind == "sweep" and "best: point" in text
