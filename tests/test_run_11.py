"""Config 11: five folds with -p fold=i over config 01 as a lower layer, then kalfa collect."""

import json
from pathlib import Path

import pytest

import kalfa  # noqa: F401
from conftest import ROOT
from kalfa.api import check, run
from kalfa.cli import main
from kalfa.collect import collect
from kalfa.config import parse_sets
from kalfa.record import read_history, read_resolved

CONFIG = str(ROOT / "configs" / "11_kfold_cv.yaml")


def test_check_folds_and_the_valid_less_form(workdir):
    prepared = check([CONFIG])
    assert prepared.problems == []
    assert prepared.sizes == {"train": 1360, "valid": 240, "test": 400}
    assert prepared.surface.data["record"] == "runs/cv_housing_0"
    no_valid = "data.split={uri: kfold, params: {k: 5, fold: 0, seed: 7}}"
    prepared = check([CONFIG], parse_sets([no_valid]))
    kinds = [problem.kind for problem in prepared.problems]
    assert kinds.count("set_missing") >= 2 and prepared.sizes == {"train": 1600, "valid": 0, "test": 400}
    prepared = check([CONFIG], parse_sets([no_valid, "training.stop=[]", "training.checkpoint=last",
                                           "training.report=last"]))
    assert [problem.kind for problem in prepared.problems] == ["set_missing"]
    assert "val/loss_mae" in prepared.problems[0].message
    prepared = check([CONFIG], parse_sets([no_valid, "training.stop=[]", "training.rules=[]",
                                           "training.checkpoint=last", "training.report=last"]))
    assert prepared.problems == []
    assert prepared.surface.data["data"]["split"] == {"uri": "/split/kalfa/kfold",
                                                       "params": {"k": 5, "fold": 0, "seed": 7}}


def test_five_folds_and_collect(workdir):
    tests = set()
    for fold in range(5):
        result = run([CONFIG], parse_sets(params=["epochs=1", f"fold={fold}"]))
        assert result.record == f"runs/cv_housing_{fold}"
        history = read_history(result.record)
        assert len(history) == 1 and "test/rmse" in history[0]
        config = read_resolved(result.record)
        assert config["params"]["fold"] == fold and config["data"]["split"]["params"]["fold"] == fold
        rows = set(__import__("pandas").read_parquet(Path(result.record) / "predictions.parquet")["row"])
        assert not rows & tests
        tests |= rows
    assert len(tests) == 2000
    kind, text, target = collect([f"runs/cv_housing_{fold}" for fold in range(5)])
    assert kind == "cv" and target == "runs"
    summary = json.loads(Path("runs/cv.json").read_text())
    assert [fold["fold"] for fold in summary["folds"]] == [0, 1, 2, 3, 4]
    assert set(summary["summary"]) >= {"test/rmse", "test/mae", "test/loss_mse"}
    assert summary["summary"]["test/rmse"]["std"] >= 0.0
    assert "| test/rmse |" in text and (Path("runs") / "cv.md").exists()
    assert main(["collect", *[f"runs/cv_housing_{fold}" for fold in range(5)]]) == 0
