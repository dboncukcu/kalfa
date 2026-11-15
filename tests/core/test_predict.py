import contextlib
from types import SimpleNamespace

import numpy
import torch
from torch import nn

from kalfa.core import predict
from kalfa.core.context import Context
from kalfa.core.wrap import Criterion, ModelEntry
from kalfa.std.steps import Supervised


CLASSES = ["cold", "mild", "hot"]


class Rows(torch.utils.data.Dataset):
    def __init__(self, fields):
        self.fields = fields

    def __len__(self):
        return len(next(iter(self.fields.values())))

    def __getitem__(self, index):
        return {name: values[index] for name, values in self.fields.items()}


class FakeData:
    inputs = ("x",)

    def __init__(self, sets, targets, label=None, classes=None, batch=16):
        self.sets = sets
        self.target_names = list(targets)
        self.label = label
        self.classes = classes
        self.batch = batch

    def loaders(self):
        return {name: torch.utils.data.DataLoader(Rows(fields), batch_size=self.batch)
                for name, fields in self.sets.items()}

    def rows(self, set_name):
        return numpy.arange(len(Rows(self.sets[set_name])))

    def rescales(self, name):
        return name != self.label

    def inverse(self, name, values):
        return numpy.asarray(values) * 10.0 + 5.0

    def is_label(self, name):
        return name == self.label


class Decoding(FakeData):
    def decode(self, name, values):
        return numpy.asarray(self.classes)[numpy.asarray(values)]


class Entry(ModelEntry):
    def call(self, batch):
        result = self.module(*[batch[name] for name in self.inputs])
        if isinstance(result, dict):
            return result
        if isinstance(result, (tuple, list)):
            return dict(zip(self.outputs, result))
        return {self.outputs[0]: result}


class Paired(Criterion):
    def pair(self, outputs, batch, ctx):
        return outputs[self.pred], batch[self.target]


class Cpu:
    type = "cpu"

    def move(self, batch):
        return batch

    def autocast(self, enabled):
        return contextlib.nullcontext()


def weather(count, seed, classes=3):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(count, 2, generator=generator)
    cuts = torch.linspace(-1.0, 1.0, classes + 1)[1:-1]
    return {"x": x, "kind": torch.bucketize(x[:, 0].contiguous(), cuts)}


def make_context(data, net, outputs, loss, target):
    torch.manual_seed(0)
    tree = SimpleNamespace(config={}, settings={"train": {"epochs": 1}})
    ctx = Context(tree)
    ctx.device = Cpu()
    ctx.data = data
    ctx.models = {"net": Entry("net", net, ["x"], outputs)}
    criterion = Paired(loss, pred=outputs[0], target=target)
    criterion.name = "loss"
    ctx.losses = {"loss": criterion}
    ctx.metrics, ctx.optimizers = {}, {}
    ctx.step = Supervised()
    return ctx


def test_multiclass_probabilities_sum_to_one_and_follow_the_softmax():
    sets = {"train": weather(32, 1), "test": weather(24, 2)}
    net = nn.Linear(2, 3)
    ctx = make_context(Decoding(sets, ["kind"], label="kind", classes=CLASSES), net, ["logits"],
                       nn.CrossEntropyLoss(), "kind")
    table = predict.predictions_table(ctx, "test")
    columns = [f"prob_logits_{name}" for name in CLASSES]
    assert list(table.columns) == ["row", "kind", "pred_logits", "raw_logits", *columns]
    probabilities = table[columns].to_numpy()
    numpy.testing.assert_allclose(probabilities.sum(axis=1), numpy.ones(24), rtol=1e-6)
    assert (probabilities >= 0).all() and (probabilities <= 1).all()
    with torch.no_grad():
        expected = torch.softmax(net(sets["test"]["x"]), dim=1).numpy()
    numpy.testing.assert_allclose(probabilities, expected, rtol=1e-5, atol=1e-7)
    assert probabilities.argmax(axis=1).tolist() == table["raw_logits"].tolist()
    assert table["pred_logits"].tolist() == [CLASSES[index] for index in table["raw_logits"]]
    assert set(table["kind"]) <= set(CLASSES)


def test_multiclass_without_decode_names_probabilities_by_index():
    sets = {"train": weather(32, 1), "test": weather(8, 2)}
    ctx = make_context(FakeData(sets, ["kind"], label="kind"), nn.Linear(2, 3), ["logits"], nn.CrossEntropyLoss(),
                       "kind")
    table = predict.predictions_table(ctx, "test")
    assert list(table.columns) == ["row", "kind", "pred_logits", "prob_logits_0", "prob_logits_1", "prob_logits_2"]
    numpy.testing.assert_allclose(table[["prob_logits_0", "prob_logits_1", "prob_logits_2"]].sum(axis=1), 1.0,
                                  rtol=1e-6)
    assert table["pred_logits"].dtype.kind == "i"


def test_single_column_output_writes_the_sigmoid_of_the_positive_class():
    sets = {"train": weather(32, 1, classes=2), "test": weather(10, 2, classes=2)}
    net = nn.Linear(2, 1)
    ctx = make_context(Decoding(sets, ["kind"], label="kind", classes=["no", "yes"]), net, ["logit"],
                       nn.BCEWithLogitsLoss(), "kind")
    table = predict.predictions_table(ctx, "test")
    assert list(table.columns) == ["row", "kind", "pred_logit", "raw_logit", "prob_logit_yes"]
    with torch.no_grad():
        logits = net(sets["test"]["x"])[:, 0]
    numpy.testing.assert_allclose(table["prob_logit_yes"].to_numpy(), torch.sigmoid(logits).numpy(), rtol=1e-5)
    assert table["raw_logit"].tolist() == (logits > 0).long().tolist()
    assert table["pred_logit"].tolist() == [["no", "yes"][index] for index in table["raw_logit"]]
    plain = make_context(FakeData(sets, ["kind"], label="kind"), net, ["logit"], nn.BCEWithLogitsLoss(), "kind")
    assert list(predict.predictions_table(plain, "test").columns) == ["row", "kind", "pred_logit", "prob_logit_1"]


def test_regression_output_has_no_probability_columns():
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(12, 2, generator=generator)
    sets = {"test": {"x": x, "price": x[:, :1] * 2}}
    ctx = make_context(FakeData(sets, ["price"]), nn.Linear(2, 1), ["net"], nn.MSELoss(), "price")
    table = predict.predictions_table(ctx, "test")
    assert list(table.columns) == ["row", "price", "pred_net", "raw_net"]
    numpy.testing.assert_allclose(table["pred_net"].to_numpy(), table["raw_net"].to_numpy() * 10 + 5, rtol=1e-5)


def test_softmax_and_sigmoid_are_stable_for_large_values():
    probabilities = predict.softmax(numpy.array([[1000.0, 0.0], [-1000.0, 0.0]]))
    numpy.testing.assert_allclose(probabilities, [[1.0, 0.0], [0.0, 1.0]], atol=1e-12)
    numpy.testing.assert_allclose(predict.sigmoid(numpy.array([0.0, 1000.0, -1000.0])), [0.5, 1.0, 0.0], atol=1e-12)
