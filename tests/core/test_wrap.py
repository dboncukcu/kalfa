import importlib
import importlib.util
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from kalfa.core.errors import BuildError, RunError
from kalfa.core.protocols import ContextMetric, Objective
from kalfa.core.wrap import (
    Criterion,
    EmaCopy,
    Group,
    MetricEntry,
    ModelEntry,
    OptimizerEntry,
    active,
    freeze_parameters,
    loss_scalar,
    wrap_loss,
    wrap_metric,
)


class Doubler(nn.Module):
    def forward(self, x):
        return x * 2


class Pair(nn.Module):
    def forward(self, x):
        return x, x + 1


class TwoPart(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3, 4)
        self.head = nn.Linear(4, 1)

    def forward(self, x):
        return self.head(torch.relu(self.backbone(x)))


class Sum(Objective):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, outputs, batch, ctx):
        return self.weight * outputs["net"].sum()


class Multiplier(Objective):
    def __init__(self):
        super().__init__()
        self.lmbda = nn.Parameter(torch.tensor(0.5))

    def forward(self, outputs, batch, ctx):
        return self.lmbda * outputs["net"].sum()


class CountRows(ContextMetric):
    def __init__(self):
        self.rows = 0

    def reset(self):
        self.rows = 0

    def update(self, outputs, batch, ctx):
        self.rows += len(batch["x"])

    def compute(self):
        return self.rows


class TimesTen:
    def inverse(self, name, values):
        assert name == "y"
        return values * 10


class MeanSquared(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("total", torch.zeros(()))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))
        self.moves = []

    def update(self, preds, target):
        if preds.shape != target.shape:
            raise RuntimeError(f"preds {tuple(preds.shape)} and target {tuple(target.shape)} differ in shape")
        self.total += ((preds - target) ** 2).sum()
        self.count += target.numel()

    def compute(self):
        return self.total / self.count

    def reset(self):
        self.total.zero_()
        self.count.zero_()

    def to(self, *args, **kwargs):
        self.moves.append(args)
        return super().to(*args, **kwargs)


class MulticlassHits(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer("hits", torch.zeros((), dtype=torch.long))
        self.register_buffer("rows", torch.zeros((), dtype=torch.long))

    def update(self, preds, target):
        if target.dtype != torch.long or preds.shape != (target.shape[0], self.num_classes):
            raise RuntimeError(f"preds {tuple(preds.shape)} and target {target.dtype} do not fit {self.num_classes} "
                               f"classes")
        self.hits += (preds.argmax(dim=1) == target).sum()
        self.rows += target.numel()

    def compute(self):
        return self.hits.float() / self.rows

    def reset(self):
        self.hits.zero_()
        self.rows.zero_()


METRIC_KINDS = ["stand_in"] + (["torchmetrics"] if importlib.util.find_spec("torchmetrics") else [])


def metric_makers(kind):
    if kind == "torchmetrics":
        torchmetrics = importlib.import_module("torchmetrics")
        return torchmetrics.MeanSquaredError, lambda: torchmetrics.classification.MulticlassAccuracy(
            num_classes=3, average="micro")
    return MeanSquared, lambda: MulticlassHits(num_classes=3)


def test_ema_shift_blends_parameters_and_copies_buffers():
    torch.manual_seed(0)
    module = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
    ema = EmaCopy(module, decay=0.75)
    before = ema.module[0].weight.detach().clone()
    with torch.no_grad():
        module[0].weight.add_(1.0)
        module[1].running_mean.fill_(3.0)
    ema.shift(module)
    assert torch.allclose(ema.module[0].weight, 0.75 * before + 0.25 * module[0].weight.detach())
    assert torch.equal(ema.module[1].running_mean, module[1].running_mean)
    assert not any(parameter.requires_grad for parameter in ema.parameters())
    assert ema.updates == 1 and not ema.training and not ema.module.training
    assert not ema.train().training
    assert ema.source is module
    assert torch.equal(ema(torch.ones(4, 2)), ema.module(torch.ones(4, 2)))


def test_ema_every_counts_calls_and_state_round_trips():
    module = nn.Linear(2, 1)
    ema = EmaCopy(module, decay=0.0, every=2)
    with torch.no_grad():
        module.weight.fill_(5.0)
    ema.shift(module)
    assert not torch.equal(ema.module.weight, module.weight)
    ema.shift(module)
    assert torch.equal(ema.module.weight, module.weight)
    state = ema.state_dict()
    assert "module.weight" in state
    other = EmaCopy(nn.Linear(2, 1), decay=0.0, every=2)
    other.load_state_dict(state)
    assert other.updates == 2
    assert torch.equal(other.module.weight, module.weight)


def test_ema_shift_accepts_a_model_entry():
    entry = ModelEntry("net", nn.Linear(2, 1), ["x"], ["net"])
    ema = EmaCopy(entry.module, decay=0.0)
    with torch.no_grad():
        entry.module.bias.fill_(2.0)
    ema.shift(entry)
    assert torch.equal(ema.module.bias, entry.module.bias)


def test_model_entry_names_outputs():
    entry = ModelEntry("net", Doubler(), ["x"], ["net"])
    out = entry.call({"x": torch.ones(2)})
    assert set(out) == {"net"} and torch.equal(out["net"], torch.full((2,), 2.0))
    pair = ModelEntry("vae", Pair(), ["x"], ["x_hat", "mu"])
    named = pair.call({"x": torch.ones(2)})
    assert list(named) == ["x_hat", "mu"] and torch.equal(named["mu"], torch.full((2,), 2.0))
    given = {"x_hat": 1, "mu": 2, "extra": 3}
    assert pair.name_outputs(given) is given
    assert pair.name_outputs([1, 2]) == {"x_hat": 1, "mu": 2}
    with pytest.raises(RunError, match="3 values"):
        pair.name_outputs((1, 2, 3))
    with pytest.raises(RunError, match="mu"):
        pair.name_outputs({"x_hat": 1})
    with pytest.raises(RunError, match="not a tensor"):
        pair.name_outputs(3)
    with pytest.raises(RunError, match=r"\['image'\]"):
        pair.call({"image": torch.ones(2)})
    assert entry.parameter_count() == 0 and entry.path == "models.net"


def test_set_freeze_patterns_true_and_false():
    entry = ModelEntry("net", TwoPart(), ["x"], ["net"], freeze=["backbone.*"])
    assert not entry.module.backbone.weight.requires_grad and not entry.module.backbone.bias.requires_grad
    assert entry.module.head.weight.requires_grad and entry.module.head.bias.requires_grad
    assert entry.frozen == ["backbone.*"] and entry.module.training and not entry.locked
    entry.set_freeze(True)
    assert not any(parameter.requires_grad for parameter in entry.module.parameters())
    assert not entry.module.training and entry.locked and entry.frozen is True
    entry.set_mode(True)
    assert not entry.module.training
    entry.set_freeze(False)
    assert all(parameter.requires_grad for parameter in entry.module.parameters())
    assert entry.frozen is False and not entry.locked
    entry.set_mode(True)
    assert entry.module.training
    entry.set_mode(False)
    assert not entry.module.training
    freeze_parameters(entry.module, "head.bias")
    assert not entry.module.head.bias.requires_grad and entry.module.head.weight.requires_grad
    freeze_parameters(entry, [])
    assert all(parameter.requires_grad for parameter in entry.module.parameters())


def test_criterion_pair_aligns_shapes_and_rescales():
    criterion = Criterion(nn.MSELoss(), pred="net", target="y")
    guess, truth = criterion.pair({"net": torch.zeros(4, 1)}, {"y": torch.ones(4)}, None)
    assert guess.shape == (4,) and truth.shape == (4,)
    guess, truth = criterion.pair({"net": torch.zeros(4)}, {"y": torch.ones(4, 1)}, None)
    assert guess.shape == (4,) and truth.shape == (4,)
    guess, truth = criterion.pair({"net": torch.zeros(4, 3)}, {"y": torch.ones(4, dtype=torch.long)}, None)
    assert guess.shape == (4, 3) and truth.shape == (4,)
    guess, truth = criterion.pair({"net": torch.zeros(4, 2)}, {"y": torch.ones(4, 2)}, None)
    assert guess.shape == (4, 2) and truth.shape == (4, 2)
    value = criterion({"net": torch.tensor([[1.0], [3.0]])}, {"y": torch.tensor([1.0, 1.0])}, None)
    assert value.item() == pytest.approx(2.0)
    scaled = Criterion(nn.MSELoss(), pred="net", target="y", rescale=True)
    with_inverse = SimpleNamespace(data=TimesTen())
    guess, truth = scaled.pair({"net": torch.ones(2, 1)}, {"y": torch.ones(2)}, with_inverse)
    assert torch.equal(guess, torch.full((2,), 10.0)) and torch.equal(truth, torch.full((2,), 10.0))
    without = SimpleNamespace(data=object())
    guess, truth = scaled.pair({"net": torch.ones(2, 1)}, {"y": torch.ones(2)}, without)
    assert torch.equal(guess, torch.ones(2))
    guess, truth = criterion.pair({"net": torch.ones(2, 1)}, {"y": torch.ones(2)}, with_inverse)
    assert torch.equal(guess, torch.ones(2))


def test_criterion_pair_errors_name_the_missing_pieces():
    unset = Criterion(nn.MSELoss())
    with pytest.raises(RunError, match="pred is not set"):
        unset.pair({"net": torch.zeros(1)}, {"y": torch.zeros(1)}, None)
    half = Criterion(nn.MSELoss(), pred="net")
    with pytest.raises(RunError, match="target is not set"):
        half.pair({"net": torch.zeros(1)}, {"y": torch.zeros(1)}, None)
    criterion = Criterion(nn.MSELoss(), pred="x_hat", target="price")
    criterion.name = "mse"
    with pytest.raises(RunError, match=r"loss 'mse'.*'x_hat'.*\['logvar', 'mu'\]"):
        criterion.pair({"mu": torch.zeros(1), "logvar": torch.zeros(1)}, {"price": torch.zeros(1)}, None)
    with pytest.raises(RunError, match=r"'price'.*\['x'\]"):
        criterion.pair({"x_hat": torch.zeros(1)}, {"x": torch.zeros(1)}, None)
    with pytest.raises(RunError, match=r"\[\]"):
        criterion.pair(None, {"price": torch.zeros(1)}, None)


def test_wrap_loss_wraps_torch_module_and_callable():
    loss = wrap_loss(nn.MSELoss(), {"pred": "net", "target": "y", "on": "train", "every": 2}, "mse", "losses.mse")
    assert isinstance(loss, Criterion) and loss.name == "mse" and loss.path == "losses.mse"
    assert loss.on == ("train",) and loss.every == 2 and loss.rescale is False
    outputs = {"net": torch.tensor([[1.0], [3.0]])}
    batch = {"y": torch.tensor([1.0, 1.0])}
    assert loss(outputs, batch, None).item() == pytest.approx(2)
    plain = wrap_loss(lambda pred, target: (pred - target).abs().sum(), {"pred": "net", "target": "y"}, "l1",
                      "losses.l1")
    assert isinstance(plain, Criterion) and plain.on == ("train", "valid", "test") and plain.every == 1
    assert plain(outputs, batch, None).item() == pytest.approx(2)
    scaled = wrap_loss(nn.L1Loss(), {"pred": "net", "target": "y", "rescale": True}, "mae", "losses.mae")
    assert scaled.rescale is True
    by_class = wrap_loss(nn.L1Loss, {"pred": "net", "target": "y"}, "mae", "losses.mae")
    assert isinstance(by_class, Criterion) and isinstance(by_class.function, nn.L1Loss)
    with pytest.raises(BuildError, match="not a loss"):
        wrap_loss(3, {}, "bad", "losses.bad")


def test_wrap_loss_keeps_objective_and_rejects_pairing_keys():
    objective = Sum()
    same = wrap_loss(objective, {"on": ["train"], "every": 3}, "total", "losses.total")
    assert same is objective and same.on == ("train",) and same.every == 3
    assert same.name == "total" and same.path == "losses.total"
    assert same({"net": torch.ones(3)}, {}, None).item() == pytest.approx(3.0)
    untouched = wrap_loss(Sum(), {}, "total", "losses.total")
    assert untouched.on == ("train", "valid", "test") and untouched.every == 1
    for key in ("pred", "target", "rescale"):
        with pytest.raises(BuildError, match=key):
            wrap_loss(Sum(), {key: "net"}, "total", "losses.total")
    criterion = Criterion(nn.L1Loss())
    again = wrap_loss(criterion, {"pred": "net", "target": "y", "rescale": True}, "mae", "losses.mae")
    assert again is criterion and again.pred == "net" and again.target == "y" and again.rescale is True


@pytest.mark.parametrize("kind", METRIC_KINDS)
def test_metric_entry_wraps_torchmetrics(kind):
    mean_squared, accuracy_metric = metric_makers(kind)
    keys = {"pred": "net", "target": "y", "rescale": False}
    entry = wrap_metric(mean_squared(), keys, "mse", "metrics.mse")
    assert isinstance(entry, MetricEntry) and entry.path == "metrics.mse" and entry.name == "mse"
    assert entry.rescale is False and entry.on == ("train", "valid", "test") and entry.every == 1
    entry.reset()
    assert math.isnan(entry.compute())
    entry.update({"net": torch.tensor([[1.0], [2.0]])}, {"y": torch.tensor([1.0, 4.0])}, None)
    entry.update({"net": torch.tensor([[0.0]])}, {"y": torch.tensor([2.0])}, None)
    assert entry.placed
    assert entry.compute() == pytest.approx(8 / 3)
    entry.reset()
    assert math.isnan(entry.compute())
    scaled = wrap_metric(mean_squared(), {"pred": "net", "target": "y"}, "mse", "metrics.mse")
    assert scaled.rescale is None
    scaled.rescale = True
    ctx = SimpleNamespace(data=TimesTen())
    scaled.update({"net": torch.tensor([[1.0], [2.0]])}, {"y": torch.tensor([1.0, 1.0])}, ctx)
    assert scaled.compute() == pytest.approx(50.0)
    accuracy = wrap_metric(accuracy_metric(), {"pred": "net", "target": "label"}, "acc", "metrics.acc")
    logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0], [2.0, 0.0, 0.0]])
    accuracy.update({"net": logits}, {"label": torch.tensor([0, 1, 2, 1], dtype=torch.int32)}, None)
    assert accuracy.compute() == pytest.approx(0.75)
    with pytest.raises(BuildError, match="not a metric"):
        wrap_metric(object(), {}, "bad", "metrics.bad")


def test_metric_entry_places_aligns_and_resets_a_torchmetrics_like_module():
    metric = MeanSquared()
    entry = wrap_metric(metric, {"pred": "net", "target": "y"}, "mse", "metrics.mse")
    assert entry.metric is metric and not entry.placed
    entry.update({"net": torch.tensor([[1.0], [3.0]])}, {"y": torch.tensor([[1.0], [1.0]])}, None)
    entry.update({"net": torch.tensor([2.0])}, {"y": torch.tensor([[2.0]])}, None)
    assert entry.placed and metric.moves == [(torch.device("cpu"),)]
    assert metric.count.item() == 3 and entry.seen == 2
    assert entry.compute() == pytest.approx(4 / 3)
    entry.reset()
    assert metric.total.item() == 0.0 and metric.count.item() == 0 and math.isnan(entry.compute())
    with pytest.raises(RuntimeError, match="differ in shape"):
        entry.update({"net": torch.ones(2, 3)}, {"y": torch.ones(2)}, None)
    by_class = wrap_metric(MeanSquared, {"pred": "net", "target": "y"}, "mse", "metrics.mse")
    assert isinstance(by_class.metric, MeanSquared)
    with pytest.raises(BuildError, match="MulticlassHits needs arguments"):
        wrap_metric(MulticlassHits, {"pred": "net", "target": "label"}, "acc", "metrics.acc")
    hits = wrap_metric(MulticlassHits(num_classes=2), {"pred": "net", "target": "label"}, "acc", "metrics.acc")
    hits.update({"net": torch.tensor([[0.0, 1.0], [1.0, 0.0]])}, {"label": torch.tensor([1, 1], dtype=torch.int16)},
                None)
    assert hits.compute() == pytest.approx(0.5)


def test_metric_entry_update_requires_pred_and_converts_dict_results():
    class Stats:
        def __init__(self):
            self.values = []

        def reset(self):
            self.values = []

        def update(self, pred, target):
            assert target.dtype == torch.long
            self.values.append(float((pred - target).abs().sum()))

        def compute(self):
            return {"sum": torch.tensor(sum(self.values)), "count": len(self.values)}

    entry = MetricEntry("stats", Stats(), pred="net", target="y")
    entry.update({"net": torch.tensor([[1.0], [2.0]])}, {"y": torch.tensor([1, 1])}, None)
    assert entry.compute() == {"sum": 1.0, "count": 1.0}
    with pytest.raises(RunError, match="metric 'stats': pred is not set"):
        MetricEntry("stats", Stats()).update({"net": torch.ones(1)}, {"y": torch.ones(1)}, None)


def test_context_metric_is_kept_as_is_and_delegated():
    rows = CountRows()
    same = wrap_metric(rows, {"on": "valid", "every": 2}, "rows", "metrics.rows")
    assert same is rows and same.on == ("valid",) and same.every == 2
    assert same.name == "rows" and same.path == "metrics.rows"
    with pytest.raises(BuildError, match="pred"):
        wrap_metric(CountRows(), {"pred": "net"}, "rows", "metrics.rows")
    entry = MetricEntry("rows", CountRows())
    entry.update(None, {"x": torch.ones(5)}, None)
    entry.update(None, {"x": torch.ones(2)}, None)
    assert entry.compute() == 7.0
    entry.reset()
    assert math.isnan(entry.compute())


def test_optimizer_groups_follow_match():
    model = TwoPart()
    model.name = "net"
    entry = OptimizerEntry("main", torch.optim.SGD, {"lr": 0.1}, [Group(model, "backbone.*", lr=0.01), model])
    assert entry.rates() == {"net[backbone.*]": 0.01} and entry.rate() == pytest.approx(0.1)
    entry.build()
    groups = entry.real.param_groups
    assert len(groups) == 2
    assert groups[0]["lr"] == pytest.approx(0.1) and groups[1]["lr"] == pytest.approx(0.01)
    assert {id(p) for p in groups[1]["params"]} == {id(p) for p in model.backbone.parameters()}
    assert {id(p) for p in groups[0]["params"]} == {id(p) for p in model.head.parameters()}
    assert entry.rate() == pytest.approx(0.1)
    assert entry.rates() == {"net[backbone.*]": pytest.approx(0.01)}
    assert len(entry.parameters()) == 4 and entry.modules() == [model]
    assert entry.base == [pytest.approx(0.1), pytest.approx(0.01)]
    assert entry.build() is entry.real
    entry.set_param("lr", 0.05)
    assert groups[0]["lr"] == pytest.approx(0.05) and groups[1]["lr"] == pytest.approx(0.01)
    assert entry.params["lr"] == 0.05 and entry.base == [pytest.approx(0.05), pytest.approx(0.01)]
    entry.set_param("momentum", 0.9)
    assert groups[0]["momentum"] == 0.9 and groups[1]["momentum"] == 0.9
    with pytest.raises(BuildError, match="nothing"):
        OptimizerEntry("main", torch.optim.SGD, {"lr": 0.1}, [Group(model, "nothing.*"), model]).build()
    with pytest.raises(BuildError, match="no parameters"):
        OptimizerEntry("main", torch.optim.SGD, {"lr": 0.1}, [None]).build()


def test_optimizer_over_accepts_entries_objectives_and_negative_group_rates():
    model = ModelEntry("net", nn.Linear(2, 1), ["x"], ["net"])
    multiplier = Multiplier()
    multiplier.name = "total"
    with torch.no_grad():
        model.module.weight.fill_(1.0)
        model.module.bias.zero_()
    entry = OptimizerEntry("main", torch.optim.Adam, {"lr": 1e-3}, [model, Group(multiplier, lr=-0.05)])
    assert entry.modules() == [model.module, multiplier]
    entry.build()
    assert entry.rates() == {"total": pytest.approx(-0.05)}
    assert len(entry.parameters()) == 3 and len(entry.real.param_groups) == 2
    before = multiplier.lmbda.item()
    entry.minimize(multiplier({"net": model.module(torch.ones(4, 2))}, {}, None))
    assert multiplier.lmbda.item() > before


def test_minimize_changes_parameters_and_shifts_ema():
    torch.manual_seed(0)
    model = nn.Linear(3, 1)
    ema = EmaCopy(model, decay=0.5)
    entry = OptimizerEntry("main", torch.optim.SGD, {"lr": 0.1}, [model])
    entry.emas = [ema]
    x = torch.randn(8, 3)
    y = torch.randn(8, 1)
    before = model.weight.detach().clone()
    norm = entry.minimize(((model(x) - y) ** 2).mean(), clip=10.0)
    assert isinstance(norm, float) and norm > 0
    assert not torch.equal(model.weight, before)
    assert entry.updates == 1 and ema.updates == 1
    assert torch.allclose(ema.module.weight, 0.5 * before + 0.5 * model.weight.detach())
    measured = entry.minimize(((model(x) - y) ** 2).mean())
    assert isinstance(measured, float) and measured > 0
    assert entry.updates == 2 and ema.updates == 2
    entry.zero_grad()
    assert all(parameter.grad is None for parameter in model.parameters())
    ((model(x) - y) ** 2).mean().backward()
    with torch.no_grad():
        model.weight.grad.fill_(3.0)
        model.bias.grad.fill_(4.0)
    assert entry.clip(1.0) == pytest.approx(math.sqrt(9 * 3 + 16))
    assert torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0).item() == pytest.approx(1.0)


def test_minimize_shifts_paired_emas_and_uses_a_scaler():
    class FakeScaler:
        def __init__(self):
            self.calls = []

        def is_enabled(self):
            return True

        def scale(self, loss):
            self.calls.append("scale")
            return loss * 2

        def unscale_(self, optimizer):
            self.calls.append("unscale")
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    parameter.grad.div_(2)

        def step(self, optimizer):
            self.calls.append("step")
            optimizer.step()

        def update(self):
            self.calls.append("update")

    model = nn.Linear(2, 1)
    ema = EmaCopy(model, decay=0.0)
    entry = OptimizerEntry("main", torch.optim.SGD, {"lr": 0.1}, [model])
    entry.emas = [ema]
    entry.scaler = FakeScaler()
    norm = entry.minimize(model(torch.ones(4, 2)).pow(2).mean(), clip=100.0)
    assert entry.scaler.calls == ["scale", "unscale", "step", "update"]
    assert isinstance(norm, float)
    assert torch.equal(ema.module.weight, model.weight)


def test_schedule_factor_scales_base_rate():
    model = nn.Linear(2, 1)
    entry = OptimizerEntry("main", torch.optim.SGD, {"lr": 0.2}, [model],
                           schedule=lambda updates: 0.5 if updates else 1.0)
    entry.build()
    assert entry.rate() == pytest.approx(0.2)
    entry.minimize(model(torch.ones(4, 2)).pow(2).mean())
    assert entry.rate() == pytest.approx(0.1)
    entry.set_param("lr", 0.4)
    assert entry.base == [pytest.approx(0.4)] and entry.rate() == pytest.approx(0.2)
    entry.set_param("schedule", None)
    assert entry.schedule is None
    entry.set_param("lr", 0.3)
    assert entry.rate() == pytest.approx(0.3)


def test_optimizer_state_round_trip_keeps_pending_until_build():
    model = nn.Linear(2, 1)
    first = OptimizerEntry("main", torch.optim.SGD, {"lr": 0.1, "momentum": 0.9}, [model])
    first.set_param("lr", 0.05)
    assert first.params["lr"] == 0.05 and first.real is None
    first.minimize(model(torch.ones(4, 2)).pow(2).mean())
    state = first.state_dict()
    assert set(state) == {"torch", "updates", "params", "base"}
    assert state["updates"] == 1 and state["base"] == [pytest.approx(0.05)] and state["params"]["lr"] == 0.05
    second = OptimizerEntry("main", torch.optim.SGD, {"lr": 0.1, "momentum": 0.9}, [model])
    second.load_state_dict(state)
    assert second.real is None and second.pending is not None and second.updates == 1
    assert second.state_dict()["torch"] is state["torch"]
    second.build()
    assert second.pending is None and second.rate() == pytest.approx(0.1) and second.base == [pytest.approx(0.1)]
    assert second.real.param_groups[0]["momentum"] == 0.9
    buffers = second.real.state_dict()["state"]
    assert buffers and all("momentum_buffer" in item for item in buffers.values())
    third = OptimizerEntry("main", torch.optim.SGD, {"lr": 0.1, "momentum": 0.9}, [model])
    third.build()
    third.load_state_dict(state)
    assert third.updates == 1 and third.rate() == pytest.approx(0.1) and third.pending is None
    third.load_state_dict(None)
    assert third.updates == 1


def test_active_and_loss_scalar():
    entry = SimpleNamespace(on=("valid",), every=2)
    assert active(entry, "valid", 2) and not active(entry, "valid", 3) and not active(entry, "train", 2)
    assert active(SimpleNamespace(), "train", 1)
    assert loss_scalar({"loss": 1}, "x") == 1
    with pytest.raises(ValueError, match="loss"):
        loss_scalar({"other": 1}, "x")
    with pytest.raises(ValueError, match="not a tensor"):
        loss_scalar(1.0, "x")
