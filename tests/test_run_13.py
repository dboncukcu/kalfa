"""Config 13: a teacher loaded from an earlier run's weights and frozen, the student distilled, lr cooled by a rule."""

from pathlib import Path

import pytest
import torch

import kalfa  # noqa: F401
from conftest import ROOT
from helpers import yaml_text
from kalfa.api import check, run
from kalfa.config import parse_sets
from kalfa.record import read_history
from kalfa.synthetic import write_image_folder

CONFIG = str(ROOT / "configs" / "13_distillation.yaml")
SMALL = ["device=cpu", "data.batch.size=16", "data.batch.eval_size=32"]
TEACHER = {
    "include": ["/alias/kalfa/vision"],
    "plugins": ["timm_legos"],
    "seed": 3,
    "data": {"source": {"uri": "image_folder", "params": {"path": "data/cifar10"}},
             "split": {"ratios": [0.9, 0.1, 0.0], "seed": 3}, "batch": {"size": 16},
             "preprocessors": {"to_tensor": {"uri": "to_tensor"},
                               "normalize": {"uri": "normalize", "params": {"mean": "cifar10", "std": "cifar10"}}},
             "fields": {"image": {"preprocessors": ["to_tensor", "normalize"]}, "label": {"target": True}},
             "feed": "table"},
    "model": {"models": {"net": {"optimizer": {"uri": "adam", "params": {"lr": 1e-3}}, "inputs": ["image"],
                                 "outputs": ["logits"],
                                 "nodes": [{"uri": "timm_backbone", "params": {"name": "resnet50", "pooled": True}},
                                           {"uri": "linear", "params": {"out_features": 10}}]}}},
    "metrics": {"accuracy": {"uri": "accuracy"}},
    "losses": {"ce": {"uri": "cross_entropy"}},
    "training": {"turn": "supervised", "loss": "ce", "epochs": 1,
                 "checkpoint": {"uri": "best", "params": {"monitor": "val/accuracy", "mode": "max"}}, "report": "best"},
    "record": "runs/cifar_resnet50_x",
}


@pytest.fixture
def cifar(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_image_folder(tmp_path / "data" / "cifar10", classes=("bird", "cat", "dog"), per_class=12, size=32, channels=3)
    (tmp_path / "teacher.yaml").write_text(yaml_text(TEACHER))
    result = run([str(tmp_path / "teacher.yaml")])
    assert result.record == "runs/cifar_resnet50_x"
    return tmp_path


def test_check_compares_the_teacher_with_its_run(cifar):
    prepared = check([CONFIG], parse_sets(SMALL))
    assert prepared.problems == []
    prepared = check([CONFIG], parse_sets(SMALL + ["model.models.teacher.weights={run: runs/nowhere, model: net, which: best}"]))
    assert "weights_run_missing" in [problem.kind for problem in prepared.problems]
    prepared = check([CONFIG], parse_sets(SMALL + ["model.models.teacher.weights={run: runs/cifar_resnet50_x, model: net, which: final}"]))
    assert "weights_missing" not in [problem.kind for problem in prepared.problems]
    prepared = check([CONFIG], parse_sets(SMALL + ["model.models.teacher.nodes=[{uri: linear, params: {out_features: 10}}]"]))
    assert "weights_mismatch" in [problem.kind for problem in prepared.problems]
    prepared = check([CONFIG], parse_sets(SMALL + ["model.models.teacher.init={weights: zeros}"]))
    assert "weights_init" in [problem.kind for problem in prepared.problems]


def test_run_13(cifar):
    teacher_state = torch.load(Path("runs/cifar_resnet50_x/checkpoints/best.pt"), weights_only=False)["models"]["net"]
    forced = ("training.rules=[{name: cool_down, when: {uri: metric_above, params: {monitor: val/accuracy, "
              "value: -1.0}}, set: {main.lr: 1.0e-4}}]")
    result = run([CONFIG], parse_sets(SMALL + [forced], ["epochs=2"]), when="fixed")
    record = Path(result.record)
    history = read_history(record)
    assert [line["turn"] for line in history] == [1, 2]
    assert {"train/kd", "train/kd/ce", "train/kd/kl", "val/kd/kl", "val/accuracy", "lr/main"} <= set(history[0])
    assert [line["rules"] for line in history] == [["cool_down"], []]
    assert history[0]["lr/main"] == pytest.approx(1e-3) and history[1]["lr/main"] == pytest.approx(1e-4)
    payload = torch.load(record / "checkpoints" / "last.pt", weights_only=False)
    teacher = payload["models"]["teacher"]
    assert set(teacher) == set(teacher_state)
    assert all(torch.equal(teacher[key], teacher_state[key]) for key in teacher)
    student = payload["models"]["student"]
    assert any(key.endswith("weight") and student[key].shape[0] == 4 for key in student)
    assert not any("teacher" in key for key in payload["optimizers"])
    assert (record / "plots" / "loss_curve.png").exists()
