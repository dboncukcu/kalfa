from types import SimpleNamespace

import numpy
import pytest
import torch
from core.test_loop import Cpu, Entry, make_context
from torch import nn

import kalfa
from helpers import housing_frame, write_parquet
from kalfa.core import build, loop, make, state
from kalfa.core import run as running
from kalfa.core.context import Context
from kalfa.core.errors import KalfaError
from kalfa.core.rules import apply_patch
from kalfa.core.wrap import EmaCopy, Group, ModelEntry, OptimizerEntry
from kalfa.records.reader import read_lines
from kalfa.std.data.table import Table
from kalfa.std.steps import Supervised


class PatchRule:
    def __init__(self, path, value, at=1):
        self.name = "extend"
        self.path = path
        self.value = value
        self.at = at

    def fire(self, ctx):
        if ctx.turn != self.at:
            return False
        apply_patch(ctx, self.path, self.value, by=self.name, turn=ctx.turn)
        return True


class TwoPart(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3, 4)
        self.head = nn.Linear(4, 1)

    def forward(self, x):
        return self.head(torch.relu(self.backbone(x)))


class Coder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 2)
        self.decoder = nn.Linear(2, 3)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def grad_norm(parameters):
    return float(torch.sqrt(sum(parameter.grad.pow(2).sum() for parameter in parameters)))


def tiny_config(file, record):
    return {
        "seed": 1,
        "device": "cpu",
        "data": {"use": "table", "file": str(file), "split": [0.6, 0.2, 0.2], "features": {"x*": "standard_scaler"},
                 "targets": {"price": "standard_scaler"}, "batch": 64},
        "models": {"net": {"use": "mlp", "hidden": [8], "out": 1}},
        "losses": {"mse": "mse"},
        "metrics": {"rmse": "rmse"},
        "optimizers": {"main": {"use": "adam", "lr": 0.01, "over": "@models.net", "minimizes": "@losses.mse"}},
        "train": {"epochs": 1, "checkpoint": {"watch": "val/rmse"}},
        "record": str(record),
    }


@pytest.mark.parametrize("settings, path, value", [
    ({"epochs": 2}, "train.epochs", 3),
    ({"updates": {"total": 6, "turn": 3}}, "train.updates.total", 9),
])
def test_a_rule_setting_the_training_length_extends_the_running_loop(tmp_path, settings, path, value):
    ctx = make_context(tmp_path, settings={**settings, "checkpoint": False})
    ctx.rules = [PatchRule(path, value)]
    history = loop.train(ctx)
    assert ctx.train.total_turns() == 3
    assert [line["turn"] for line in history] == [1, 2, 3]
    assert ctx.record.notes["turns_planned"] == 3


def test_the_update_counter_counts_optimizer_steps_under_accumulate(tmp_path):
    ctx = make_context(tmp_path, settings={"epochs": 1, "checkpoint": False})
    ctx.step = Supervised(accumulate=2)
    loop.train(ctx)
    steps = ctx.optimizers["main"].calls["step"]
    assert steps == 2
    assert ctx.updates == steps
    assert len(read_lines(ctx.record.directory / "updates.jsonl")) == steps
    assert ctx.history[0]["update"] == steps


def test_clip_measures_and_clips_only_the_parameters_the_optimizer_owns():
    torch.manual_seed(0)
    model = TwoPart()
    entry = OptimizerEntry("head", torch.optim.SGD, {"lr": 0.1}, [Group(model, "head.*")])
    entry.build()
    owned = [parameter for group in entry.real.param_groups for parameter in group["params"]]
    assert {id(parameter) for parameter in owned} == {id(parameter) for parameter in model.head.parameters()}
    entry.zero_grad()
    model(torch.randn(8, 3)).pow(2).mean().backward()
    expected = grad_norm(owned)
    assert grad_norm(model.parameters()) > expected
    assert entry.clip(None) == pytest.approx(expected)
    norm = entry.clip(0.5 * expected)
    assert norm == pytest.approx(expected)
    assert grad_norm(owned) == pytest.approx(0.5 * expected, rel=1e-4)
    entry.zero_grad()
    assert all(parameter.grad is None for parameter in model.head.parameters())
    assert all(parameter.grad is not None for parameter in model.backbone.parameters())


def test_resume_remembers_the_best_of_the_source_run(tmp_path):
    source = make_context(tmp_path / "source", settings={"epochs": 2, "checkpoint": {"watch": "val/mae"}})
    loop.train(source)
    assert source.best is not None
    resumed = make_context(tmp_path / "resumed", settings={"epochs": 3, "checkpoint": {"watch": "val/mae"}})
    directory = source.record.directory
    state.load_checkpoint(resumed, directory / "checkpoints" / "last.pt")
    running.carry_lines(directory, resumed, resumed.turn)
    assert resumed.turn == 2 and [line["turn"] for line in resumed.history] == [1, 2]
    assert resumed.best == source.best
    resumed.turn += 1
    resumed.latest = {"val/mae": source.best + 1.0}
    assert "best" not in loop.checkpoint_tags(resumed)
    resumed.latest = {"val/mae": source.best - 1.0}
    assert "best" in loop.checkpoint_tags(resumed)


def test_predict_frame_aligns_the_frame_columns_when_the_filter_drops_rows(tmp_path):
    file = write_parquet(housing_frame(160, seed=11), tmp_path / "housing.parquet")
    table = Table(str(file), split=[0.6, 0.2, 0.2], features={"x*": "standard_scaler"},
                  targets={"price": "standard_scaler"}, filter="price > 190")
    table.fit(SimpleNamespace(config=None, seed=0, rng=None))
    ctx = Context(SimpleNamespace(config={}, settings={"epochs": 1}))
    ctx.device = Cpu()
    ctx.data = table
    entry = Entry("net", nn.Linear(4, 1), ["x"], ["net"])
    ctx.models = {"net": entry}
    ctx.step = Supervised()
    fresh = housing_frame(40, seed=12)
    kept = fresh[fresh["price"] > 190]
    assert 0 < len(kept) < len(fresh)
    found = running.predict_frame(ctx, table.loader_for(fresh), fresh, entry)
    assert list(found.columns) == ["row", "price", "pred_net", "raw_net"]
    assert len(found) == len(kept)
    assert found["row"].tolist() == kept.index.tolist()
    numpy.testing.assert_allclose(found["price"].to_numpy(), kept["price"].to_numpy())
    plain = Table(str(file), split=[0.6, 0.2, 0.2], features={"x*": "standard_scaler"},
                  targets={"price": "standard_scaler"})
    plain.fit(SimpleNamespace(config=None, seed=0, rng=None))
    ctx.data = plain
    unlabeled = fresh.drop(columns=["price"])
    blind = running.predict_frame(ctx, plain.loader_for(unlabeled), unlabeled, entry)
    assert list(blind.columns) == ["row", "pred_net", "raw_net"]
    assert blind["row"].tolist() == fresh.index.tolist()
    labeled = running.predict_frame(ctx, plain.loader_for(fresh), fresh, entry)
    numpy.testing.assert_allclose(blind["pred_net"].to_numpy(), labeled["pred_net"].to_numpy())


def test_resuming_into_the_source_record_refuses_or_continues_without_duplicating_its_lines(tmp_path):
    file = write_parquet(housing_frame(120), tmp_path / "housing.parquet")
    record = tmp_path / "record"
    kalfa.run(tiny_config(file, record))
    assert [line["turn"] for line in read_lines(record / "history.jsonl")] == [1]
    try:
        running.resume(record, sets=[("train.epochs", 2)], target=record)
    except KalfaError:
        assert [line["turn"] for line in read_lines(record / "history.jsonl")] == [1]
    else:
        assert [line["turn"] for line in read_lines(record / "history.jsonl")] == [1, 2]
        updates = read_lines(record / "updates.jsonl")
        assert [line["update"] for line in updates] == list(range(1, len(updates) + 1))


def test_resume_carries_the_best_checkpoint_and_serves_it_for_report_best(tmp_path):
    file = write_parquet(housing_frame(120), tmp_path / "housing.parquet")
    record = tmp_path / "record"
    config = tiny_config(file, record)
    config["train"]["report"] = "best"
    kalfa.run(config)
    target = tmp_path / "resumed"
    result = running.resume(record, sets=[("train.epochs", 2)], target=target)
    assert result.context.turn == 2
    assert (target / "checkpoints" / "best.pt").exists()
    assert [line["turn"] for line in read_lines(target / "history.jsonl")] == [1, 2]
    assert [event["kind"] for event in read_lines(target / "events.jsonl")][-1] == "run_end"


def test_ema_shifts_when_the_optimizer_covers_a_submodule_of_its_model():
    module = Coder()
    entry = ModelEntry("ae", module, ["x"], ["x_hat"])
    entry.ema = EmaCopy(module, decay=0.5)
    ctx = SimpleNamespace(models={"ae": entry})
    optimizer = OptimizerEntry("dec", torch.optim.SGD, {"lr": 0.1}, [module.decoder])
    make.link_emas(optimizer, [module.decoder], ctx)
    optimizer.build()
    optimizer.minimize(module(torch.ones(4, 3)).pow(2).mean())
    assert entry.ema.updates == 1


def test_a_batchnorm_frozen_by_pattern_keeps_its_running_statistics():
    module = nn.Sequential(nn.BatchNorm1d(3), nn.Linear(3, 1))
    entry = ModelEntry("net", module, ["x"], ["net"], freeze=["0.*"])
    entry.set_mode(True)
    before = module[0].running_mean.clone()
    module(torch.randn(16, 3) + 5.0)
    assert module[1].training
    assert torch.equal(module[0].running_mean, before)
    entry.set_freeze(False)
    entry.set_mode(True)
    assert module[0].training
