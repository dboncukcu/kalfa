from types import SimpleNamespace

import pytest
import torch
from torch import nn

from helpers import housing_frame, write_parquet
from kalfa.core import build, check as checking, loop, record, rules, run as running, spec, state
from kalfa.core.context import Context
from kalfa.core.errors import BuildError
from kalfa.core.protocols import Step
from kalfa.core.wrap import OptimizerEntry


class Idle(Step):
    def train_turn(self, ctx):
        return {}


class Extend:
    name = "extend"

    def fire(self, ctx):
        if ctx.turn != 1:
            return False
        rules.apply_patch(ctx, "train.epochs", 4, by=self.name)
        return True


def bare_context(train):
    tree = SimpleNamespace(config={}, settings={"train": train}, nodes={}, rules={})
    ctx = Context(tree)
    ctx.step = Idle()
    ctx.data = SimpleNamespace(loaders=lambda: {})
    ctx.train.evaluate = []
    return ctx


def test_a_rule_patching_train_epochs_extends_the_run():
    ctx = bare_context({"epochs": 2})
    ctx.rules = [Extend()]
    loop.train(ctx)
    assert ctx.train.epochs == 4
    assert ctx.turn == 4
    assert [line["rules"] for line in ctx.history] == [["extend"], [], [], []]


def test_checkpoint_counts_the_applied_patches():
    ctx = bare_context({"epochs": 2})
    rules.apply_patch(ctx, "train.epochs", 3, by="longer", turn=1)
    assert state.payload(ctx)["patches"] == 1
    assert [patch.path for patch in ctx.patches] == ["train.epochs"]


def test_resolved_yaml_keeps_the_plugins_key():
    config = spec.load(mappings=[{"plugins": ["myplug"], "seed": 1, "train": {"epochs": 1}}])
    assert config.plugins == ["myplug"]
    text = record.resolved_text(config)
    assert text.startswith("plugins:") and "myplug" in text
    assert "seed: 1" in text
    bare = spec.load(mappings=[{"seed": 1}])
    assert bare.plugins == [] and "plugins" not in record.resolved_text(bare)


def test_plan_rule_patch_values_are_written_as_references():
    mapping = {
        "data": {"use": "table", "file": "housing.parquet", "split": [0.7, 0.15, 0.15],
                 "features": {"x*": "standard_scaler"}, "targets": {"price": "standard_scaler"}},
        "models": {"net": {"use": "mlp", "hidden": [8], "out": 1}},
        "losses": {"mse": "mse", "huber": {"use": "huber", "delta": 1.0}},
        "optimizers": {"main": {"use": "adam", "lr": 1.0e-3, "over": "@models.net", "minimizes": "@losses.mse"}},
        "train": {"epochs": 2},
        "rules": {"to_huber": {"when": {"use": "after_epoch", "at": 1},
                               "set": {"optimizers.main.minimizes": "@losses.huber"}}},
    }
    checked = checking.check(mappings=[mapping])
    assert checked.ok, checked.text()
    assert checked.plan["nodes"]["optimizers.main"]["keys"]["minimizes"] == {"ref": "losses.mse"}
    assert checked.plan["rules"][0]["patch"][0]["value"] == {"ref": "losses.huber"}
    assert checked.plan["rules"][0]["when"] == {"use": "after_epoch", "at": 1}


def test_data_reference_outside_exposes_is_refused():
    tree = SimpleNamespace(nodes={})
    ctx = SimpleNamespace(data=SimpleNamespace(exposes=("n_classes",), n_classes=3, batch={"size": 7}))
    ref = build.Ref("@data.batch", "data", "batch", "models.net.out")
    with pytest.raises(BuildError, match="exposes"):
        build.locate(tree, ctx, ref)
    assert build.locate(tree, ctx, build.Ref("@data.n_classes", "data", "n_classes", "models.net.out")) == 3


def test_check_refuses_a_data_reference_outside_exposes():
    mapping = {
        "data": {"use": "table", "file": "housing.parquet", "split": [0.7, 0.15, 0.15],
                 "targets": {"price": "standard_scaler"}},
        "models": {"net": {"use": "mlp", "hidden": [8], "out": "@data.batch"}},
        "losses": {"mse": "mse"},
        "optimizers": {"main": {"use": "adam", "lr": 1.0e-3, "over": "@models.net", "minimizes": "@losses.mse"}},
        "train": {"epochs": 1},
    }
    checked = checking.check(mappings=[mapping])
    assert [(problem.code, problem.path) for problem in checked.errors] == [("reference_unknown", "models.net")]


def test_config_lr_survives_loading_an_optimizer_checkpoint():
    net = nn.Linear(2, 1)
    old = OptimizerEntry("main", torch.optim.SGD, {"lr": 0.01}, [net])
    old.build()
    saved = old.state_dict()
    new = OptimizerEntry("main", torch.optim.SGD, {"lr": 0.5}, [net])
    new.build()
    new.load_state_dict(saved)
    assert new.rate() == pytest.approx(0.5)
    assert new.base == [pytest.approx(0.5)]


def test_a_loss_may_reference_a_loss_declared_after_it(tmp_path):
    path = write_parquet(housing_frame(), tmp_path / "housing.parquet")
    mapping = {
        "device": "cpu",
        "data": {"use": "table", "file": str(path), "split": [0.7, 0.15, 0.15],
                 "features": {"x*": "standard_scaler"}, "targets": {"price": "standard_scaler"}},
        "models": {"net": {"use": "mlp", "hidden": [8], "out": 1}},
        "losses": {"total": {"use": "weighted_sum", "terms": [{"of": "@losses.mse", "weight": 1.0}]}, "mse": "mse"},
        "optimizers": {"main": {"use": "adam", "lr": 1.0e-3, "over": "@models.net", "minimizes": "@losses.total"}},
        "train": {"epochs": 1},
    }
    checked = checking.check(mappings=[mapping])
    assert checked.ok, checked.text()
    ctx = running.prepare(checked, record=False)
    assert list(ctx.losses) == ["total", "mse"]
    assert ctx.losses["total"].terms[0].of is ctx.losses["mse"]


def test_a_reference_cycle_inside_a_section_is_an_error(tmp_path):
    path = write_parquet(housing_frame(), tmp_path / "housing.parquet")
    mapping = {
        "device": "cpu",
        "data": {"use": "table", "file": str(path), "split": [0.7, 0.15, 0.15],
                 "features": {"x*": "standard_scaler"}, "targets": {"price": "standard_scaler"}},
        "models": {"net": {"use": "mlp", "hidden": [8], "out": 1}},
        "losses": {"a": {"use": "weighted_sum", "terms": [{"of": "@losses.b"}]},
                   "b": {"use": "weighted_sum", "terms": [{"of": "@losses.a"}]}},
        "optimizers": {"main": {"use": "adam", "lr": 1.0e-3, "over": "@models.net", "minimizes": "@losses.a"}},
        "train": {"epochs": 1},
    }
    checked = checking.check(mappings=[mapping])
    assert checked.ok, checked.text()
    with pytest.raises(BuildError, match="cycle"):
        running.prepare(checked, record=False)
