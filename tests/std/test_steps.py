import pytest
from core.test_loop import Entry, Sgd, install_checkpoints, make_context
from torch import nn

from kalfa.core import aliases, loop
from kalfa.core.errors import RunError
from kalfa.core.protocols import Objective
from kalfa.records.reader import read_lines
from kalfa.std.steps import Alternating, Supervised


class Counting(nn.Linear):
    def __init__(self, *shape):
        super().__init__(*shape)
        self.forwards = 0

    def forward(self, value):
        self.forwards += 1
        return super().forward(value)


class Direct(Objective):
    def __init__(self, model, sign=1.0):
        super().__init__()
        self.model = model
        self.sign = sign

    def forward(self, outputs, batch, ctx):
        return self.sign * self.model(batch["x"]).mean()


@pytest.fixture
def saved(monkeypatch):
    install_checkpoints(monkeypatch)


def first_batch(ctx):
    ctx.batches += 1
    return next(iter(ctx.loaders()["train"]))


def two_optimizers(ctx):
    net = ctx.models["net"].module
    ctx.optimizers = {"a": Sgd("a", net, ctx.losses["mse"]), "b": Sgd("b", net, ctx.losses["mae"])}
    return ctx.optimizers


def test_supervised_minimizes_in_order_and_caches_outputs():
    ctx = make_context()
    net = Counting(2, 1)
    ctx.models = {"net": Entry("net", net, ["x"], ["net"])}
    ctx.optimizers = {"b": Sgd("b", net, ctx.losses["mae"]), "a": Sgd("a", net, ctx.losses["mse"])}
    step = Supervised(order=["a", "b"], grad_clip=1.0)
    batch = first_batch(ctx)
    result = step.train_batch(batch, ctx)
    assert list(result) == ["a", "b"] and all(value.requires_grad for value in result.values())
    assert ctx.optimizers["a"].calls["minimize"] == 1 and ctx.optimizers["b"].calls["minimize"] == 1
    assert ctx.last_norms["a"] > 0 and ctx.last_norms["b"] > 0
    assert net.forwards == 2
    outputs = step.eval_batch(batch, ctx)
    assert net.forwards == 2 and not outputs["net"].requires_grad
    ctx.batches += 1
    step.eval_batch(batch, ctx)
    assert net.forwards == 3
    step.predict(batch, ctx)
    assert net.forwards == 4
    single = make_context()
    only = Counting(2, 1)
    single.models = {"net": Entry("net", only, ["x"], ["net"])}
    single.optimizers = {"main": Sgd("main", only, single.losses["mse"])}
    Supervised().train_batch(first_batch(single), single)
    assert only.forwards == 1


def test_supervised_without_a_predicting_model_passes_empty_outputs():
    ctx = make_context()
    ctx.models = {}
    ctx.losses = {"direct": Direct(ctx.optimizers["main"].module)}
    ctx.losses["direct"].name = "direct"
    ctx.optimizers["main"].minimizes = ctx.losses["direct"]
    step = Supervised()
    batch = first_batch(ctx)
    assert set(step.train_batch(batch, ctx)) == {"main"}
    assert step.eval_batch(batch, ctx) == {} and step.predict(batch, ctx) == {}


def test_supervised_order_must_name_optimizers():
    ctx = make_context()
    with pytest.raises(RunError, match="order names optimizers"):
        Supervised(order=["ghost"]).train_batch(first_batch(ctx), ctx)


def test_supervised_rejects_a_loss_without_gradient():
    ctx = make_context()
    for parameter in ctx.models["net"].module.parameters():
        parameter.requires_grad_(False)
    with pytest.raises(RunError, match="carries no gradient"):
        Supervised().train_batch(first_batch(ctx), ctx)


def test_supervised_rejects_an_optimizer_without_a_loss():
    ctx = make_context()
    ctx.optimizers["main"].minimizes = None
    with pytest.raises(RunError, match="minimizes nothing"):
        Supervised().train_batch(first_batch(ctx), ctx)


def test_supervised_accumulates_gradients():
    ctx = make_context()
    step = Supervised(accumulate=2, grad_clip=5.0)
    optimizer = ctx.optimizers["main"]
    norms = []
    for batch in ctx.loaders()["train"]:
        ctx.batches += 1
        step.train_batch(batch, ctx)
        norms.append(ctx.last_norms["main"])
    assert optimizer.calls == {"zero_grad": 2, "backward": 4, "step": 2, "minimize": 0}
    assert [norm is None for norm in norms] == [True, False, True, False]


def test_alternating_repeats_on_the_shared_batch(tmp_path):
    ctx = make_context(tmp_path, settings={"epochs": 1, "checkpoint": False})
    two_optimizers(ctx)
    ctx.step = Alternating(order=["a", "b"], repeat={"a": 2})
    found = ctx.step.train_turn(ctx)
    assert set(found) == {"mse", "mae", "abs"}
    assert ctx.optimizers["a"].calls["minimize"] == 8 and ctx.optimizers["b"].calls["minimize"] == 4
    assert ctx.updates == 4 and ctx.batches == 4
    lines = read_lines(ctx.record.directory / "updates.jsonl")
    assert len(lines) == 4 and set(lines[0]) == {"update", "turn", "loss/a", "lr/a", "loss/b", "lr/b"}
    assert [line["update"] for line in lines] == [1, 2, 3, 4]


def test_alternating_takes_fresh_batches():
    ctx = make_context(sizes=(192, 32, 32), settings={"epochs": 1})
    two_optimizers(ctx)
    ctx.step = Alternating(order=["a", "b"], repeat={"a": 2, "b": 1}, fresh_batch=True)
    ctx.step.train_turn(ctx)
    assert ctx.optimizers["a"].calls["minimize"] == 4 and ctx.optimizers["b"].calls["minimize"] == 2
    assert ctx.updates == 2 and ctx.batches == 6


def test_alternating_without_a_predicting_model():
    ctx = make_context(settings={"epochs": 1})
    generator, critic = nn.Linear(2, 2), nn.Linear(2, 1)
    ctx.models = {"g": Entry("g", generator, ["x"], ["fake"]), "d": Entry("d", critic, ["x"], ["score"])}
    ctx.losses = {"d": Direct(critic), "g": Direct(generator, sign=-1.0)}
    for name, objective in ctx.losses.items():
        objective.name = name
    ctx.metrics = {}
    ctx.optimizers = {"d": Sgd("d", critic, ctx.losses["d"]), "g": Sgd("g", generator, ctx.losses["g"])}
    ctx.step = Alternating(order=["d", "g"], repeat={"d": 3, "g": 1})
    found = ctx.step.train_turn(ctx)
    assert set(found) == {"d", "g"}
    assert ctx.optimizers["d"].calls["minimize"] == 12 and ctx.optimizers["g"].calls["minimize"] == 4


def test_alternating_through_train_writes_history(tmp_path, saved):
    ctx = make_context(tmp_path, settings={"epochs": 2})
    two_optimizers(ctx)
    ctx.step = Alternating(order=["a", "b"], repeat={"a": 2})
    history = loop.train(ctx)
    assert len(history) == 2 and ctx.updates == 8
    line = history[-1]
    assert line["update"] == 8 and line["minimizes/a"] == "mse" and line["minimizes/b"] == "mae"
    assert {"train/mse", "train/mae", "train/abs", "val/mse", "val/mae", "val/abs"} <= set(line)
    assert (ctx.record.directory / "checkpoints" / "last.pt").exists()


def test_alternating_by_updates_with_a_turn_of_three_updates():
    ctx = make_context(settings={"updates": {"total": 6, "turn": 3}})
    two_optimizers(ctx)
    ctx.step = Alternating(order=["a", "b"], repeat={"a": 2})
    history = loop.train(ctx)
    assert [line["update"] for line in history] == [3, 6]
    assert ctx.optimizers["a"].calls["minimize"] == 12 and ctx.optimizers["b"].calls["minimize"] == 6


def test_steps_are_registered_when_the_alias_table_works():
    assert "supervised" in aliases.ALIASES and "alternating" in aliases.ALIASES
