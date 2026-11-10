import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from helpers import housing_frame
from kalfa.commands import sweep as sweeping
from kalfa.core.check import check
from kalfa.core.errors import BuildError
from kalfa.records import reader
from records.fixtures import make_record


SPACE = {"lr": {"log": [0.0001, 0.1]}, "width": {"int": [2, 16]}}


def sweep_config(home, root, strategy, space):
    return {
        "seed": 3,
        "device": "cpu",
        "params": {"lr": 0.01, "width": 8},
        "data": {"use": "table", "file": str(home / "housing.parquet"), "split": [0.7, 0.15, 0.15],
                 "features": {"x*": "standard_scaler"}, "targets": {"price": "standard_scaler"}, "batch": 32},
        "models": {"net": {"use": "mlp", "hidden": ["$width$"], "out": 1}},
        "losses": {"mse": "mse"},
        "metrics": {"rmse": "rmse"},
        "optimizers": {"main": {"use": "adam", "lr": "$lr$", "over": "@models.net", "minimizes": "@losses.mse"}},
        "train": {"epochs": 1, "checkpoint": {"watch": "val/rmse"}, "report": "best"},
        "sweep": {"strategy": strategy, "space": space,
                  "objective": {"watch": "val/rmse", "mode": "min", "at": "best"}, "record": str(root)},
    }


@pytest.fixture(scope="module")
def home(tmp_path_factory):
    root = tmp_path_factory.mktemp("phase3_review")
    housing_frame(160, 0).to_parquet(root / "housing.parquet", index=False)
    return root


def fake_attempt(plan, index, values, monitor):
    directory = Path(sweeping.point_dir(plan.root, index))
    make_record(directory, params=dict(values), sweep={"root": plan.root, "id": index, "values": dict(values)},
                seed=index)
    return SimpleNamespace(record=str(directory), context=None)


def test_p3_r1_optuna_strategy_refuses_to_build_without_the_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "optuna", None)
    with pytest.raises(BuildError, match=r"optuna needs the optuna package \(uv sync installs the dependencies of "
                                         r"kalfa\)"):
        sweeping.Optuna(trials=2, seed=1)


def test_p3_r1_check_reports_an_optuna_strategy_without_the_package(home, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "optuna", None)
    config = sweep_config(home, tmp_path / "asked", {"use": "optuna", "trials": 2, "seed": 1}, SPACE)
    codes = [problem.code for problem in check(mappings=[config]).problems]
    assert "sweep_strategy" in codes


def test_p3_r2_an_optuna_sweep_that_goes_on_asks_new_points(home, tmp_path, monkeypatch):
    root = tmp_path / "optuna"
    config = sweep_config(home, root, {"use": "optuna", "trials": 4, "sampler": "random", "seed": 1}, SPACE)
    monkeypatch.setattr(sweeping, "attempt", fake_attempt)
    first = sweeping.plan_of(mappings=[config])
    first.strategy.trials = 2
    assert len(sweeping.run_local(first, log=lambda text: None)) == 2
    second = sweeping.plan_of(mappings=[config])
    assert len(sweeping.run_local(second, log=lambda text: None)) == 2
    values = [line["values"] for line in reader.read_lines(root / "points.jsonl")]
    assert [point["id"] for point in reader.read_lines(root / "points.jsonl")] == [0, 1, 2, 3]
    assert values[2] != values[0] and values[3] != values[1]
