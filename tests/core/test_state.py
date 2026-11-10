import logging
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from kalfa.core.errors import KalfaError
from kalfa.core.rng import Rng
from kalfa.core.state import checkpoint_path, load_checkpoint, payload, save_checkpoint, stateful_nodes


class FakeEntry:
    def __init__(self, module, ema=None):
        self.module = module
        self.ema = ema

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state):
        self.module.load_state_dict(state)


class FakeOptimizer:
    def __init__(self, lr):
        self.lr = lr
        self.updates = 0

    def state_dict(self):
        return {"lr": self.lr, "updates": self.updates}

    def load_state_dict(self, state):
        self.lr = state["lr"]
        self.updates = state["updates"]


class Buffered(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("lmbda", torch.tensor(0.5))

    def forward(self, outputs, batch, ctx):
        return self.lmbda


class FakeStep:
    def __init__(self):
        self.phase = 0

    def state_dict(self):
        return {"phase": self.phase}

    def load_state_dict(self, state):
        self.phase = state["phase"]


class FakeRule:
    def __init__(self, name):
        self.name = name
        self.fired_turn = None

    def state_dict(self):
        return {"fired_turn": self.fired_turn}

    def load_state_dict(self, state):
        self.fired_turn = state["fired_turn"]


def make_ctx(seed=7):
    torch.manual_seed(seed)
    return SimpleNamespace(
        models={"net": FakeEntry(nn.Linear(3, 1), ema=nn.Linear(3, 1)), "aux": FakeEntry(nn.Linear(2, 2))},
        losses={"mse": nn.MSELoss(), "total": Buffered()},
        metrics={"rmse": SimpleNamespace(state_dict=lambda: {}, load_state_dict=lambda state: None)},
        optimizers={"main": FakeOptimizer(0.001)},
        step=FakeStep(),
        rules=[FakeRule("stop"), FakeRule("to_huber")],
        rng=Rng(seed),
        turn=0,
        updates=0,
        patches=[],
        best=None,
        logger=logging.getLogger("kalfa.test"),
    )


def advanced_ctx():
    ctx = make_ctx()
    ctx.turn = 2
    ctx.updates = 40
    ctx.optimizers["main"].lr = 0.0005
    ctx.optimizers["main"].updates = 40
    ctx.step.phase = 3
    ctx.rules[0].fired_turn = 1
    ctx.losses["total"].lmbda.fill_(0.9)
    ctx.rng.normal("noise", (3,))
    ctx.best = 0.25
    return ctx


def test_stateful_nodes_order():
    paths = [path for path, _ in stateful_nodes(make_ctx())]
    assert paths == ["models.net", "models.net.ema", "models.aux", "losses.total", "optimizers.main", "train.step",
                     "rules.stop", "rules.to_huber"]
    bare = SimpleNamespace(models={"net": FakeEntry(nn.Linear(1, 1))}, losses={}, metrics={}, optimizers={},
                           step=SimpleNamespace(state_dict=lambda: {}), rules=[])
    assert [path for path, _ in stateful_nodes(bare)] == ["models.net"]


def test_payload_shape():
    ctx = make_ctx()
    data = payload(ctx)
    assert set(data) == {"format", "turn", "update", "nodes", "rng", "patches", "best"}
    assert data["format"] == 1 and data["turn"] == 0 and data["update"] == 0 and data["patches"] == 0
    assert data["best"] is None
    assert set(data["nodes"]) == {path for path, _ in stateful_nodes(ctx)}
    assert data["rng"]["seed"] == 7
    ctx.patches = [1, 2]
    assert payload(ctx)["patches"] == 2


def test_save_and_load_round_trip(tmp_path):
    ctx = advanced_ctx()
    path = save_checkpoint(ctx, tmp_path / "checkpoints" / "last.pt")
    assert path.exists() and [p.name for p in path.parent.iterdir()] == ["last.pt"]
    expected_noise = ctx.rng.normal("noise", (3,))
    weight = ctx.models["net"].module.weight.detach().clone()
    ema_weight = ctx.models["net"].ema.weight.detach().clone()
    other = make_ctx(seed=99)
    assert not torch.equal(other.models["net"].module.weight, weight)
    data = load_checkpoint(other, path)
    assert data["format"] == 1 and data["turn"] == 2
    assert torch.equal(other.models["net"].module.weight, weight)
    assert torch.equal(other.models["net"].ema.weight, ema_weight)
    assert torch.equal(other.models["aux"].module.weight, ctx.models["aux"].module.weight)
    assert other.optimizers["main"].lr == 0.0005 and other.optimizers["main"].updates == 40
    assert other.step.phase == 3
    assert other.rules[0].fired_turn == 1 and other.rules[1].fired_turn is None
    assert float(other.losses["total"].lmbda) == pytest.approx(0.9)
    assert other.turn == 2 and other.updates == 40 and other.best == 0.25
    assert other.rng.seed == 7
    assert torch.equal(other.rng.normal("noise", (3,)), expected_noise)


def test_only_counters_and_rng_flags(tmp_path):
    ctx = advanced_ctx()
    path = save_checkpoint(ctx, tmp_path / "last.pt")
    other = make_ctx(seed=99)
    aux_before = other.models["aux"].module.weight.detach().clone()
    load_checkpoint(other, path, only=["models.net"], counters=False, rng=False)
    assert torch.equal(other.models["net"].module.weight, ctx.models["net"].module.weight)
    assert torch.equal(other.models["net"].ema.weight, ctx.models["net"].ema.weight)
    assert torch.equal(other.models["aux"].module.weight, aux_before)
    assert other.optimizers["main"].lr == 0.001 and other.step.phase == 0
    assert other.turn == 0 and other.updates == 0 and other.best is None
    assert other.rng.streams == {} and other.rng.pending == {} and other.rng.seed == 99
    third = make_ctx(seed=99)
    load_checkpoint(third, path, only="optimizers")
    assert third.optimizers["main"].lr == 0.0005 and third.step.phase == 0 and third.turn == 2


def test_missing_node_is_skipped_with_a_debug_line(tmp_path, caplog):
    ctx = advanced_ctx()
    path = save_checkpoint(ctx, tmp_path / "last.pt")
    other = make_ctx()
    del other.models["aux"]
    other.rules = []
    with caplog.at_level(logging.DEBUG, logger="kalfa.test"):
        load_checkpoint(other, path)
    assert "models.aux" in caplog.text and "rules.stop" in caplog.text
    assert other.step.phase == 3


def test_foreign_file_is_rejected(tmp_path):
    path = tmp_path / "old.pt"
    torch.save({"version": 1, "components": {}}, path)
    with pytest.raises(KalfaError):
        load_checkpoint(make_ctx(), path)


def test_checkpoint_path(tmp_path):
    with pytest.raises(KalfaError):
        checkpoint_path(tmp_path)
    base = tmp_path / "checkpoints"
    base.mkdir()
    (base / "final.pt").write_bytes(b"")
    assert checkpoint_path(tmp_path) == base / "final.pt"
    with pytest.raises(KalfaError):
        checkpoint_path(tmp_path, "last")
    (base / "last.pt").write_bytes(b"")
    assert checkpoint_path(tmp_path) == base / "last.pt"
    assert checkpoint_path(tmp_path, "final") == base / "final.pt"
    with pytest.raises(KalfaError):
        checkpoint_path(tmp_path, "best")
    with pytest.raises(KalfaError):
        checkpoint_path(tmp_path, "bogus")
