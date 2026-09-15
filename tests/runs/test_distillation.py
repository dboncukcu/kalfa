from pathlib import Path

import pytest
import torch
from runs.conftest import SMALL

from kalfa.api import check, run
from kalfa.config import parse_sets
from kalfa.std.common.history import History


pytestmark = pytest.mark.slow

SETS, PARAMS = SMALL["13_distillation"]


@pytest.fixture
def teacher(dataset, records):
    dataset("13_distillation")
    if "teacher" not in records:
        result = run(["teacher.yaml"], parse_sets(params=["epochs=1"]))
        assert result.record == "runs/cifar_resnet50_x"
        records["teacher"] = result
    return records["teacher"]


def test_check_compares_the_teacher_with_its_run(teacher):
    prepared = check(["config.yaml"], parse_sets(SETS))
    assert prepared.problems == []
    prepared = check(["config.yaml"], parse_sets(SETS + ["model.models.teacher.weights={run: runs/nowhere, model: net, "
                                                        "which: best}"]))
    assert "weights_run_missing" in [problem.kind for problem in prepared.problems]
    prepared = check(["config.yaml"], parse_sets(SETS + ["model.models.teacher.weights={run: runs/cifar_resnet50_x, "
                                                        "model: net, which: final}"]))
    assert "weights_missing" not in [problem.kind for problem in prepared.problems]
    prepared = check(["config.yaml"], parse_sets(SETS + ["model.models.teacher.nodes=[{uri: linear, "
                                                        "params: {out_features: 10}}]"]))
    assert "weights_mismatch" in [problem.kind for problem in prepared.problems]
    prepared = check(["config.yaml"], parse_sets(SETS + ["model.models.teacher.init={weights: zeros}"]))
    assert "weights_init" in [problem.kind for problem in prepared.problems]


def test_the_frozen_teacher_and_the_cooled_student(teacher):
    teacher_state = torch.load(Path("runs/cifar_resnet50_x/checkpoints/best.pt"), weights_only=False)["models"]["net"]
    forced = ("training.rules=[{name: cool_down, when: {uri: metric_above, params: {monitor: val/accuracy, "
              "value: -1.0}}, set: {main.lr: 1.0e-4}}]")
    result = run(["config.yaml"], parse_sets(SETS + [forced], PARAMS), when="fixed")
    record = Path(result.record)
    history = History.read(record)
    assert [line["turn"] for line in history] == [1, 2]
    assert {"train/kd", "train/kd/ce", "train/kd/kl", "val/kd/kl", "val/accuracy", "lr/main"} <= set(history[0])
    assert [line["rules"] for line in history] == [["cool_down"], []]
    assert history[0]["lr/main"] == pytest.approx(1e-3) and history[1]["lr/main"] == pytest.approx(1e-4)
    payload = torch.load(record / "checkpoints" / "last.pt", weights_only=False)
    teacher_weights = payload["models"]["teacher"]
    assert set(teacher_weights) == set(teacher_state)
    assert all(torch.equal(teacher_weights[key], teacher_state[key]) for key in teacher_weights)
    student = payload["models"]["student"]
    assert any(key.endswith("weight") and student[key].shape[0] == 4 for key in student)
    assert not any("teacher" in key for key in payload["optimizers"])
    assert (record / "plots" / "loss_curve.png").exists()
