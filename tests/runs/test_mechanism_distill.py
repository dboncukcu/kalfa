from pathlib import Path

import pandas
import pytest
import torch

from helpers import config_path
from kalfa.api import check
from kalfa.config import parse_sets
from kalfa.std.checkpoint.base import load
from kalfa.std.common.history import History


pytestmark = pytest.mark.slow


@pytest.fixture
def teacher(trained, at_root):
    return trained("teacher")


@pytest.fixture
def distill(teacher, trained):
    return trained("distill")


def test_the_teacher_record_decodes_its_labels_and_draws_the_confusion_matrix(teacher):
    assert teacher.record == "runs/teacher"
    table = pandas.read_parquet(Path(teacher.record) / "predictions.parquet")
    assert list(table.columns) == ["row", "churned", "raw_logits_0", "raw_logits_1", "pred_logits"]
    assert set(table["pred_logits"]) <= {"yes", "no"} and set(table["churned"]) == {"yes", "no"}
    assert sorted(path.name for path in (Path(teacher.record) / "plots").iterdir()) == ["confusion.png"]


def test_the_weights_reference_is_checked_against_the_source_run(teacher):
    assert check([config_path("distill")], parse_sets([])).problems == []
    cases = [("model.models.teacher.weights.run=runs/nowhere", "weights_run_missing"),
             ("model.models.teacher.nodes=[{uri: linear, params: {out_features: 2}}]", "weights_mismatch"),
             ("model.models.teacher.init={weights: {uri: zeros}}", "weights_init")]
    for setting, kind in cases:
        found = check([config_path("distill")], parse_sets([setting]))
        assert kind in [problem.kind for problem in found.errors], setting
    final = check([config_path("distill")], parse_sets(["model.models.teacher.weights.which=final"]))
    assert final.problems == []


def test_the_frozen_teacher_keeps_the_loaded_weights_and_takes_no_optimizer(teacher, distill):
    last = load(Path(distill.record) / "checkpoints" / "last.pt")
    source = load(Path(teacher.record) / "checkpoints" / "best.pt")["models"]["net"]
    assert list(last["models"]) == ["teacher", "student"]
    for key, value in last["models"]["teacher"].items():
        assert torch.equal(value, source[key]), key
    assert list(last["optimizers"]) == ["main"]


def test_the_distillation_terms_and_the_cooling_rule_reach_the_history(distill):
    history = History.read(distill.record)
    terms = [f"{prefix}/{name}" for prefix in ("train", "val", "test") for name in ("kd", "kd/ce", "kd/kl", "accuracy")]
    assert list(history[0]) == ["turn", "global_step", *terms, "lr/main", "minimizes/main", "seconds", "rules"]
    assert list(history[1]) == ["turn", "global_step", *terms, "lr/main", "minimizes/main", "effect/main.lr",
                                "seconds", "rules"]
    assert [line["rules"] for line in history] == [["cool"], []]
    assert [line["lr/main"] for line in history] == [1e-3, 1e-4]
    table = pandas.read_parquet(Path(distill.record) / "predictions.parquet")
    assert list(table.columns) == ["row", "churned", "raw_logits_0", "raw_logits_1", "pred_logits"]
    assert sorted(path.name for path in (Path(distill.record) / "plots").iterdir()) == ["confusion.png",
                                                                                       "loss_curve.png"]
