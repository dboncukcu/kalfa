import contextlib
import json
from types import SimpleNamespace

import numpy
import pandas
import pytest
import torch
from torch import nn

from kalfa.core import loop, state
from kalfa.core.context import Context, TrainSettings
from kalfa.core.errors import RunError
from kalfa.core.history import Means, history_line, prefixed
from kalfa.core.predict import predictions_table
from kalfa.core.protocols import ContextMetric
from kalfa.core.record import RecordWriter
from kalfa.core.wrap import Criterion, ModelEntry, OptimizerEntry
from kalfa.records.reader import read_lines
from kalfa.std.steps import Supervised


CLASSES = ["cold", "mild", "hot"]


class Rows(torch.utils.data.Dataset):
    def __init__(self, fields):
        self.fields = fields

    def __len__(self):
        return len(next(iter(self.fields.values())))

    def __getitem__(self, index):
        return {name: values[index] for name, values in self.fields.items()}


def housing(count, seed):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(count, 2, generator=generator)
    noise = 0.1 * torch.randn(count, 1, generator=generator)
    return {"x": x, "price": 2 * x[:, :1] - x[:, 1:] + noise}


def weather(count, seed):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(count, 2, generator=generator)
    return {"x": x, "kind": torch.bucketize(x[:, 0].contiguous(), torch.tensor([-0.5, 0.5]))}


class FakeData:
    inputs = ("x",)

    def __init__(self, sets, targets, batch=32, label=None, spectators=None):
        self.sets = sets
        self.target_names = list(targets)
        self.batch = batch
        self.label = label
        self.extra = spectators or {}

    def loaders(self):
        return {name: torch.utils.data.DataLoader(Rows(fields), batch_size=self.batch)
                for name, fields in self.sets.items()}

    def rows(self, set_name):
        return numpy.arange(len(Rows(self.sets[set_name]))) + 1000

    def rescales(self, name):
        return name != self.label

    def inverse(self, name, values):
        return numpy.asarray(values) * 10.0 + 5.0

    def decode(self, name, values):
        return numpy.asarray(CLASSES)[numpy.asarray(values)]

    def spectators(self, set_name):
        return self.extra.get(set_name, {})


class Entry(ModelEntry):
    def call(self, batch):
        return self.name_outputs(self.module(*[batch[name] for name in self.inputs]))

    def name_outputs(self, result):
        if isinstance(result, dict):
            return result
        if isinstance(result, (tuple, list)):
            return dict(zip(self.outputs, result))
        return {self.outputs[0]: result}


class Paired(Criterion):
    def pair(self, outputs, batch, ctx):
        return outputs[self.pred], batch[self.target]


class Sgd(OptimizerEntry):
    def __init__(self, name, module, minimizes, lr=0.05):
        super().__init__(name, torch.optim.SGD, {"lr": lr}, [module], minimizes)
        self.module = module
        self.real = torch.optim.SGD(module.parameters(), lr=lr)
        self.calls = {"zero_grad": 0, "backward": 0, "step": 0, "minimize": 0}

    def zero_grad(self):
        self.calls["zero_grad"] += 1
        self.real.zero_grad(set_to_none=True)

    def backward(self, loss):
        self.calls["backward"] += 1
        loss.backward()

    def clip(self, limit):
        if limit is None:
            return None
        return float(torch.nn.utils.clip_grad_norm_(self.module.parameters(), float(limit)))

    def step(self):
        self.calls["step"] += 1
        self.real.step()
        self.updates += 1

    def minimize(self, loss, clip=None):
        self.calls["minimize"] += 1
        self.zero_grad()
        self.backward(loss)
        norm = self.clip(clip)
        self.step()
        return norm

    def rate(self):
        return float(self.real.param_groups[0]["lr"])

    def rates(self):
        return {}

    def state_dict(self):
        return self.real.state_dict()

    def load_state_dict(self, state):
        self.real.load_state_dict(state)


class MeanAbs(ContextMetric):
    def __init__(self, pred="net", target="price", on=("train", "valid", "test"), every=1):
        self.pred = pred
        self.target = target
        self.on = tuple(on)
        self.every = every
        self.total = 0.0
        self.count = 0

    def reset(self):
        self.total = 0.0
        self.count = 0

    def update(self, outputs, batch, ctx):
        difference = (outputs[self.pred] - batch[self.target]).abs()
        self.total += float(difference.sum())
        self.count += difference.numel()

    def compute(self):
        return self.total / max(self.count, 1)


class Record(RecordWriter):
    def __init__(self, directory):
        super().__init__(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.notes = {}

    def manifest(self, **changes):
        self.notes.update(changes)
        self.write_json("manifest.json", self.notes)
        return self.notes


class Cpu:
    type = "cpu"

    def move(self, batch):
        return batch

    def autocast(self, enabled):
        return contextlib.nullcontext()


class Halt:
    def __init__(self, name="halt", at=1):
        self.name = name
        self.at = at

    def fire(self, ctx):
        if ctx.turn != self.at:
            return False
        ctx.stop(f"rule {self.name}")
        return True


def make_context(tmp_path=None, settings=None, sizes=(128, 32, 32), seed=0):
    torch.manual_seed(seed)
    tree = SimpleNamespace(config={}, settings={"train": settings if settings is not None else {"epochs": 2}})
    ctx = Context(tree)
    ctx.device = Cpu()
    town = numpy.array(["ank", "ist"] * (sizes[2] // 2))
    sets = {"train": housing(sizes[0], 1), "valid": housing(sizes[1], 2), "test": housing(sizes[2], 3)}
    ctx.data = FakeData(sets, ["price"], spectators={"test": {"town": town}})
    net = nn.Linear(2, 1)
    ctx.models = {"net": Entry("net", net, ["x"], ["net"])}
    mse = Paired(nn.MSELoss(), pred="net", target="price")
    mse.name = "mse"
    mae = Paired(nn.L1Loss(), pred="net", target="price")
    mae.name = "mae"
    ctx.losses = {"mse": mse, "mae": mae}
    ctx.metrics = {"abs": MeanAbs()}
    ctx.optimizers = {"main": Sgd("main", net, mse)}
    ctx.step = Supervised()
    if tmp_path is not None:
        ctx.record = Record(tmp_path / "run")
    return ctx


def install_checkpoints(monkeypatch):
    def payload(ctx):
        nodes = {f"models.{name}": {key: value.clone() for key, value in entry.state_dict().items()}
                 for name, entry in ctx.models.items()}
        return {"format": 1, "turn": ctx.turn, "update": ctx.updates, "nodes": nodes}

    def load_checkpoint(ctx, path, only=None, counters=True, rng=True):
        data = state.read_checkpoint(path)
        for name, entry in ctx.models.items():
            entry.load_state_dict(data["nodes"][f"models.{name}"])
        return data

    monkeypatch.setattr(state, "payload", payload)
    monkeypatch.setattr(state, "load_checkpoint", load_checkpoint)


@pytest.fixture
def checkpoints(monkeypatch):
    install_checkpoints(monkeypatch)


def test_predicts_named_single_and_none():
    ctx = make_context(settings={"epochs": 1, "predicts": "@models.net"})
    assert ctx.predicts() is ctx.models["net"]
    ctx.train.predicts = None
    assert ctx.predicts() is ctx.models["net"]
    ctx.models["other"] = Entry("other", nn.Linear(2, 1), ["x"], ["other"])
    assert ctx.predicts() is None
    ctx.train.predicts = "@models.missing"
    with pytest.raises(RunError, match="missing"):
        ctx.predicts()


def test_set_modes_keeps_frozen_entries_in_eval():
    ctx = make_context()
    ctx.models["frozen"] = Entry("frozen", nn.Linear(2, 1), ["x"], ["frozen"], freeze=True)
    ctx.set_modes(True)
    assert ctx.models["net"].module.training and ctx.losses["mse"].training
    assert not ctx.models["frozen"].module.training
    ctx.set_modes(False)
    assert not ctx.models["net"].module.training and not ctx.losses["mse"].training


def test_train_settings_from_mapping():
    settings = TrainSettings.from_mapping({"epochs": "3", "checkpoint": {"watch": "val/mae"}, "evaluate": "valid",
                                           "track": "losses.mse.weight"})
    assert settings.epochs == 3 and settings.evaluate == ["valid"] and settings.report == "last"
    assert settings.checkpoint == {"watch": "val/mae"} and settings.track == ["losses.mse.weight"]
    assert settings.unit == "epochs" and settings.total_turns() == 3
    by_updates = TrainSettings.from_mapping({"updates": {"total": 7, "turn": 3}, "report": "best"})
    assert by_updates.unit == "updates" and by_updates.total_turns() == 3 and by_updates.report == "best"
    assert TrainSettings.from_mapping(None).evaluate == ["valid", "test"]
    assert TrainSettings.from_mapping({"evaluate": None}).evaluate == []


def test_means_records_terms_weighted_by_size():
    means = Means()
    means.record("elbo", {"loss": torch.tensor(1.0), "kl": 0.5}, 10)
    means.record("elbo", {"loss": torch.tensor(4.0), "kl": 2.0}, 30)
    means.record("mse", 2.0, 5)
    assert means.result() == pytest.approx({"elbo": 3.25, "elbo/kl": 1.625, "mse": 2.0})


def test_prefixed_uses_set_prefixes():
    merged = prefixed({"train": {"mse": 1.0}, "valid": {"mse": 2.0}, "test": {"mse": 3.0}, "holdout": {"mse": 4.0}})
    assert merged == {"train/mse": 1.0, "val/mse": 2.0, "test/mse": 3.0, "holdout/mse": 4.0}


def test_history_line_shape():
    optimizer = SimpleNamespace(rate=lambda: 0.01, rates=lambda: {"backbone": 0.001},
                                minimizes=SimpleNamespace(name="mse"))
    ctx = SimpleNamespace(turn=3, updates=12, optimizers={"main": optimizer}, fired=["to_huber"])
    line = history_line(ctx, {"train/mse": 0.5, "val/rmse": 0.7}, 1.23456)
    assert line == {"turn": 3, "update": 12, "train/mse": 0.5, "val/rmse": 0.7, "lr/main": 0.01,
                    "lr/main/backbone": 0.001, "minimizes/main": "mse", "seconds": 1.235, "rules": ["to_huber"]}


def test_train_two_turns_writes_history_updates_events_and_checkpoints(tmp_path, checkpoints):
    ctx = make_context(tmp_path, settings={"epochs": 2, "checkpoint": {"watch": "val/mae"}})
    history = loop.train(ctx)
    assert len(history) == 2 and ctx.turn == 2 and ctx.updates == 8
    first, last = history
    assert first["turn"] == 1 and first["update"] == 4 and last["turn"] == 2 and last["update"] == 8
    for key in ("train/mse", "train/mae", "train/abs", "val/mse", "val/mae", "val/abs", "test/mse", "test/abs"):
        assert key in first
    assert first["minimizes/main"] == "mse" and first["lr/main"] == pytest.approx(0.05)
    assert first["rules"] == [] and first["seconds"] >= 0
    assert last["train/mse"] < first["train/mse"]
    directory = ctx.record.directory
    assert read_lines(directory / "history.jsonl") == json.loads(json.dumps(history))
    updates = read_lines(directory / "updates.jsonl")
    assert [line["update"] for line in updates] == list(range(1, 9))
    assert [line["turn"] for line in updates] == [1] * 4 + [2] * 4
    assert set(updates[0]) == {"update", "turn", "loss/main", "lr/main"}
    kinds = [event["kind"] for event in read_lines(directory / "events.jsonl")]
    assert kinds == ["run_start", "checkpoint", "turn", "checkpoint", "turn"]
    assert (directory / "checkpoints" / "last.pt").exists() and (directory / "checkpoints" / "best.pt").exists()
    assert ctx.best == pytest.approx(min(first["val/mae"], last["val/mae"]))
    assert state.read_checkpoint(directory / "checkpoints" / "last.pt")["turn"] == 2
    assert json.loads((directory / "manifest.json").read_text())["turn"] == 2


def test_rules_fire_after_evaluation_and_can_stop(tmp_path, checkpoints):
    ctx = make_context(tmp_path, settings={"epochs": 5})
    ctx.rules = [Halt("halt", at=2)]
    history = loop.train(ctx)
    assert [line["rules"] for line in history] == [[], ["halt"]]
    assert ctx.stopping == "rule halt" and ctx.turn == 2
    events = read_lines(ctx.record.directory / "events.jsonl")
    assert events[-2]["kind"] == "turn" and events[-2]["rules"] == ["halt"]
    assert events[-1]["kind"] == "stop" and events[-1]["reason"] == "rule halt" and events[-1]["turn"] == 2
    loop.after(ctx)
    events = read_lines(ctx.record.directory / "events.jsonl")
    assert events[-1]["kind"] == "run_end" and events[-1]["stopping"] == "rule halt"


def test_stop_json_ends_the_run_after_the_turn(tmp_path):
    ctx = make_context(tmp_path, settings={"epochs": 3, "checkpoint": False})
    ctx.record.write_json("stop.json", {"by": "tester"})
    loop.train(ctx)
    assert ctx.turn == 1 and ctx.stopping == "stop.json by tester"
    kinds = [event["kind"] for event in read_lines(ctx.record.directory / "events.jsonl")]
    assert kinds == ["run_start", "turn", "stop"]


def test_checkpoint_every_and_last_false(tmp_path, checkpoints):
    ctx = make_context(tmp_path, settings={"epochs": 2, "checkpoint": {"last": False, "every": 2}})
    loop.train(ctx)
    assert sorted(path.name for path in (ctx.record.directory / "checkpoints").iterdir()) == ["turn_2.pt"]


def test_training_by_updates_keeps_a_cursor_across_turns():
    ctx = make_context(settings={"updates": {"total": 7, "turn": 3}})
    history = loop.train(ctx)
    assert len(history) == 3 and ctx.updates == 7 and ctx.batches > 7
    assert [line["update"] for line in history] == [3, 6, 7]
    assert ctx.cursor is not None and not ctx.cursor.exhausted


def test_evaluate_honours_on_every_and_skips_missing_or_empty_sets():
    ctx = make_context()
    ctx.data.sets["empty"] = {"x": torch.zeros(0, 2), "price": torch.zeros(0, 1)}
    assert loop.evaluate(ctx, "nowhere") == {}
    assert loop.evaluate(ctx, "empty") == {}
    assert set(loop.evaluate(ctx, "valid")) == {"mse", "mae", "abs"}
    ctx.losses["mae"].on = ("valid",)
    ctx.metrics["abs"].every = 2
    ctx.turn = 1
    assert set(loop.evaluate(ctx, "test")) == {"mse"}
    ctx.turn = 2
    assert set(loop.evaluate(ctx, "test")) == {"mse", "abs"}


def test_train_without_train_batches_is_an_error():
    ctx = make_context(settings={"epochs": 1})
    ctx.data.sets["train"] = {"x": torch.zeros(0, 2), "price": torch.zeros(0, 1)}
    with pytest.raises(RunError, match="no batches"):
        loop.train(ctx)


def test_tracked_reads_live_values():
    track = ["losses.mse.weight", "optimizers.main.lr", "losses.mse.terms[1].epsilon", "losses.mse.lmbda",
             "train.epochs"]
    ctx = make_context(settings={"epochs": 1, "track": track})
    ctx.losses["mse"].weight = 0.5
    ctx.losses["mse"].terms = [SimpleNamespace(epsilon=0.1), SimpleNamespace(epsilon=0.02)]
    ctx.losses["mse"].lmbda = torch.tensor([1.0, 2.0])
    assert loop.tracked(ctx) == {"track/losses.mse.weight": 0.5, "track/optimizers.main.lr": 0.05,
                                 "track/losses.mse.terms[1].epsilon": 0.02, "track/losses.mse.lmbda[0]": 1.0,
                                 "track/losses.mse.lmbda[1]": 2.0, "track/train.epochs": 1.0}
    ctx.train.track = ["losses.nope.weight"]
    with pytest.raises(RunError, match="no loss named 'nope'"):
        loop.tracked(ctx)
    ctx.train.track = ["losses.mse.nothing"]
    with pytest.raises(RunError, match="has no attribute 'nothing'"):
        loop.tracked(ctx)


def test_after_writes_final_summary_predictions_samples_and_plots(tmp_path, checkpoints):
    ctx = make_context(tmp_path, settings={"epochs": 2, "checkpoint": {"watch": "val/mae"}, "report": "best"})
    loop.train(ctx)
    drawn = []

    def curve(record, figures, series=None):
        drawn.append((record, figures.directory, series))

    ctx.plots = {"loss_curve": curve, "with_params": (curve, {"series": ["train/mse"]})}
    ctx.generate = SimpleNamespace(n=4, sample=lambda ctx, n: torch.zeros(n, 2))
    tables = loop.after(ctx)
    directory = ctx.record.directory
    assert (directory / "checkpoints" / "final.pt").exists()
    best = state.read_checkpoint(directory / "checkpoints" / "best.pt")["nodes"]["models.net"]["weight"]
    assert torch.equal(ctx.models["net"].module.weight.detach(), best)
    summary = json.loads((directory / "model_summary.json").read_text())
    assert summary["net"]["parameters"] == 3 and summary["net"]["trainable"] == 3
    assert summary["net"]["class"].endswith(".Linear") and summary["net"]["tree"] == []
    assert summary["net"]["outputs"] == {"net": [32, 1]}
    table = pandas.read_parquet(directory / "predictions.parquet")
    assert list(table.columns) == ["row", "price", "pred_net", "raw_net", "town"]
    assert len(table) == 32 and table["row"].tolist() == list(range(1000, 1032))
    truth = ctx.data.sets["test"]["price"].numpy()[:, 0] * 10 + 5
    numpy.testing.assert_allclose(table["price"].to_numpy(), truth, rtol=1e-5)
    numpy.testing.assert_allclose(table["pred_net"].to_numpy(), table["raw_net"].to_numpy() * 10 + 5, rtol=1e-5)
    assert table["town"].tolist()[:2] == ["ank", "ist"]
    numpy.testing.assert_allclose(tables["test"]["pred_net"].to_numpy(), table["pred_net"].to_numpy())
    assert torch.load(directory / "samples" / "samples.pt", weights_only=True).shape == (4, 2)
    assert [item[2] for item in drawn] == [None, ["train/mse"]]
    assert drawn[0][0] == directory and drawn[0][1] == directory / "plots"
    assert json.loads((directory / "manifest.json").read_text())["status"] == "done"
    assert read_lines(directory / "events.jsonl")[-1]["kind"] == "run_end"


def test_report_best_without_best_checkpoint_is_an_error(tmp_path, checkpoints):
    ctx = make_context(tmp_path, settings={"epochs": 1, "report": "best"})
    loop.train(ctx)
    with pytest.raises(RunError, match="best.pt"):
        loop.after(ctx)


def test_predictions_table_decodes_labels_and_needs_a_predicting_model():
    ctx = make_context()
    ctx.data = FakeData({"train": weather(64, 1), "test": weather(32, 2)}, ["kind"], label="kind")
    net = nn.Linear(2, 3)
    ctx.models = {"net": Entry("net", net, ["x"], ["logits"])}
    ctx.losses, ctx.metrics, ctx.optimizers = {}, {}, {}
    table = predictions_table(ctx, "test")
    assert list(table.columns) == ["row", "kind", "pred_logits", "raw_logits", "prob_logits_cold", "prob_logits_mild",
                                   "prob_logits_hot"]
    expected = [CLASSES[index] for index in ctx.data.sets["test"]["kind"].tolist()]
    assert table["kind"].tolist() == expected
    with torch.no_grad():
        indexes = net(ctx.data.sets["test"]["x"]).argmax(dim=1).tolist()
    assert table["raw_logits"].tolist() == indexes
    assert table["pred_logits"].tolist() == [CLASSES[index] for index in indexes]
    ctx.models = {}
    assert predictions_table(ctx, "test").empty
