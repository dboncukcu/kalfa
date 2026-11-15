import functools
import json
from pathlib import Path

import numpy
import pandas
import pytest
import torch
from torch import nn
from torch.nn import functional

import kalfa
from kalfa.commands.guide.training import criterion_notes
from kalfa.core.errors import BuildError, RunError
from kalfa.core.protocols import Objective
from kalfa.core.rules import holder_with
from kalfa.core.wrap import Criterion, wrap_loss
from kalfa.records.reader import read_lines
from kalfa.std.data import table
from kalfa.std.data.weights import weights_of
from kalfa.std.data.window import Window
from kalfa.std.losses import cross_entropy


RULES = {"price < 5": 5, "5 <= price < 15": 12}
PARAMS = {"split": [0.6, 0.2, 0.2], "features": {"x*": "standard_scaler"}, "targets": {"price": "standard_scaler"},
          "drop": ["city"], "spectators": ["id"], "batch": 16}


class FakeRng:
    def stream(self, name, device=None):
        return torch.Generator().manual_seed(11)


class FakeContext:
    def __init__(self, seed=7):
        self.seed = seed
        self.rng = FakeRng()


class Total(Objective):
    def forward(self, outputs, batch, ctx):
        return outputs["net"].sum()


def priced_frame(rows=100, seed=0):
    rng = numpy.random.default_rng(seed)
    frame = pandas.DataFrame({"x1": rng.normal(size=rows), "x2": rng.normal(size=rows)})
    frame["price"] = rng.permutation(numpy.linspace(0.0, 20.0, rows))
    frame["city"] = rng.choice(["a", "b"], size=rows)
    frame["id"] = numpy.arange(1000, 1000 + rows)
    frame["ev"] = rng.uniform(0.5, 2.0, size=rows)
    return frame


def expected(price):
    price = numpy.asarray(price, dtype="float64")
    return numpy.where(price < 5, 5.0, numpy.where(price < 15, 12.0, 1.0))


@pytest.fixture
def priced(tmp_path):
    frame = priced_frame()
    path = tmp_path / "priced.parquet"
    frame.to_parquet(path)
    return path, frame


def fitted(path, **overrides):
    data = table.Table(str(path), **{**PARAMS, **overrides})
    data.fit(FakeContext())
    return data


def test_conditions_weigh_raw_rows_and_average_one_on_train(priced):
    path, frame = priced
    data = fitted(path, weights={"w": RULES})
    raw = {name: expected(frame.loc[data.rows(name), "price"]) for name in data.counts}
    scale = 1.0 / raw["train"].mean()
    assert data.weight_scales == {"w": pytest.approx(scale)}
    for name, values in raw.items():
        held = data.datasets[name].targets["w"]
        assert held.dtype == torch.float32 and tuple(held.shape) == (data.counts[name],)
        numpy.testing.assert_allclose(held.numpy(), values * scale, rtol=1e-6)
    assert float(data.datasets["train"].targets["w"].mean()) == pytest.approx(1.0, rel=1e-6)
    assert set(numpy.unique(raw["train"])) == {1.0, 5.0, 12.0}
    assert set(data.datasets["train"][0]) == {"x", "price", "w"}
    batch = next(iter(data.loaders()["valid"]))
    assert tuple(batch["w"].shape) == (16,) and tuple(batch["price"].shape) == (16, 1)
    assert data.target_names == ["price"] and data.feature_names == ["x1", "x2"]


def test_the_first_condition_a_row_meets_wins(priced):
    path, frame = priced
    data = fitted(path, weights={"w": {"price < 15": 2, "price < 5": 9}})
    price = frame.loc[data.rows("test"), "price"].to_numpy()
    raw = data.datasets["test"].targets["w"].numpy() / data.weight_scales["w"]
    numpy.testing.assert_allclose(raw, numpy.where(price < 15, 2.0, 1.0), rtol=1e-6)


def test_a_weight_column_is_read_and_is_never_a_feature(priced):
    path, frame = priced
    data = fitted(path, weights={"w": "ev"}, features={"*": None})
    assert data.feature_names == ["x1", "x2"]
    rows = data.rows("train")
    scale = 1.0 / frame.loc[rows, "ev"].mean()
    numpy.testing.assert_allclose(data.datasets["train"].targets["w"].numpy(), frame.loc[rows, "ev"] * scale,
                                  rtol=1e-6)
    same = fitted(path, weights={"ev": "ev"})
    assert "ev" in same.datasets["train"][0] and same.feature_names == ["x1", "x2"]


def test_a_condition_reads_dropped_and_spectator_columns(priced):
    path, frame = priced
    data = fitted(path, weights={"w": {"city == 'a' and id >= 1050": 3}})
    rows = data.rows("test")
    picked = (frame.loc[rows, "city"] == "a") & (frame.loc[rows, "id"] >= 1050)
    raw = data.datasets["test"].targets["w"].numpy() / data.weight_scales["w"]
    numpy.testing.assert_allclose(raw, numpy.where(picked, 3.0, 1.0), rtol=1e-6)


def test_weight_specs_are_checked(priced):
    path, frame = priced
    assert weights_of(None) == []
    for spec, match in (({"x": RULES}, "named x"), ({"price": RULES}, "is a target"), ({"city": RULES}, "in drop"),
                        ({"w": {}}, "weights.w"), ({"w": 3}, "weights.w"), ([1], "weights maps"),
                        ({"w": {"price < 5": "heavy"}}, "finite number"), ({"w": {"price < 5": True}}, "finite number"),
                        ({"w": {5: 2}}, "condition")):
        with pytest.raises(BuildError, match=match):
            table.Table(str(path), **{**PARAMS, "weights": spec})
    for spec, match in (({"ev": RULES}, "already has a column"), ({"w": {"prize < 5": 2}}, "failed"),
                        ({"w": {"price + 1": 2}}, "true or false"), ({"w": "weight"}, "no column"),
                        ({"w": "city"}, "not numeric"), ({"w": {"price >= 0": 0}}, "positive"),
                        ({"w": {"price < 5": -1, "price >= 5": -2}}, "positive")):
        with pytest.raises(BuildError, match=match):
            fitted(path, weights=spec, drop=[])
    holes = frame.copy()
    holes.loc[3, "ev"] = numpy.nan
    holes.to_parquet(path)
    with pytest.raises(BuildError, match="finite"):
        fitted(path, weights={"w": "ev"})


def test_the_scale_is_saved_and_new_rows_get_no_weight(priced, tmp_path):
    path, frame = priced
    data = fitted(path, weights={"w": RULES})
    record = tmp_path / "record"
    data.save(record)
    fields = json.loads((record / "fitted" / "fields.json").read_text())
    assert fields["weights"] == [{"name": "w", "scale": pytest.approx(data.weight_scales["w"])}]
    loaded = table.Table(str(path), **{**PARAMS, "weights": {"w": RULES}}).load(record)
    loaded.rebuild(FakeContext())
    assert torch.equal(loaded.datasets["test"].targets["w"], data.datasets["test"].targets["w"])
    fresh = frame[["x1", "x2"]].head(8)
    assert set(next(iter(loaded.loader_for(fresh)))) == {"x"}
    unfitted = table.Table(str(path), **{**PARAMS, "weights": {"w": RULES}})
    unfitted.fitted = True
    with pytest.raises(BuildError, match="was not fitted"):
        unfitted.weight_tensors(unfitted.weighed(frame))


def test_the_report_and_the_flowchart_show_the_weights(priced):
    path, frame = priced
    data = fitted(path, weights={"w": RULES, "e": "ev"})
    report = data.report()
    assert report["weights"]["w"] == {"from": {"price < 5": 5.0, "5 <= price < 15": 12.0},
                                      "scale": pytest.approx(data.weight_scales["w"])}
    assert report["weights"]["e"]["from"] == "ev"
    assert "weights" not in fitted(path).report()
    nodes = {node["id"]: node for node in data.pipeline()["nodes"]}
    assert nodes["weights:w"]["kind"] == "weights" and "else 1" in nodes["weights:w"]["detail"]
    assert nodes["weights:e"]["detail"] == "column ev" and nodes["tensor:w"]["kind"] == "tensor"


def test_a_window_takes_the_weights_of_its_horizon_rows(tmp_path):
    frame = pandas.DataFrame({"f1": numpy.arange(40.0), "y": numpy.arange(40.0) * 10.0})
    path = tmp_path / "series.parquet"
    frame.to_parquet(path)
    data = table.Table(str(path), split={"use": "sequential", "ratios": [0.5, 0.25, 0.25]}, features={"f*": None},
                       targets={"y": None}, feed=Window(3, horizon=2), weights={"w": {"y >= 100": 4}}, batch=4)
    data.fit(FakeContext())
    scale = data.weight_scales["w"]
    for name in data.datasets:
        for batch in data.loaders()[name]:
            assert tuple(batch["w"].shape) == tuple(batch["y"].shape) == (len(batch["x"]), 2)
            wanted = torch.where(batch["y"] >= 100, 4.0, 1.0) * scale
            assert torch.allclose(batch["w"], wanted)


def test_a_streamed_table_weighs_every_chunk(tmp_path):
    frame = priced_frame(rows=90)
    path = tmp_path / "rows.parquet"
    frame.to_parquet(path, index=False)
    data = table.Table(str(path), split={"use": "sequential", "ratios": [0.6, 0.2, 0.2]},
                       features={"x*": "standard_scaler"}, targets={"price": None}, drop=["city", "ev"],
                       spectators=["id"], batch=8, stream={"chunk": 16, "fit_rows": 1000}, weights={"w": RULES})
    data.fit(FakeContext())
    train = expected(frame["price"].to_numpy()[:54])
    assert data.weight_scales["w"] == pytest.approx(1.0 / train.mean())
    assert data.counts == {"train": 54, "valid": 18, "test": 18} and data.feature_names == ["x1", "x2"]
    seen = 0
    for name in ("train", "valid"):
        for batch in data.loaders()[name]:
            wanted = expected(batch["price"].reshape(-1).numpy()) * data.weight_scales["w"]
            numpy.testing.assert_allclose(batch["w"].numpy(), wanted, rtol=1e-6)
            seen += len(wanted)
    assert seen == 72


def weighted_loss(function, **keys):
    return wrap_loss(function, {"pred": "net", "target": "y", "sample_weight": "w", **keys}, "mse", "losses.mse")


def test_a_criterion_multiplies_the_loss_of_every_row():
    outputs = {"net": torch.tensor([[1.0], [3.0], [0.0]], requires_grad=True)}
    batch = {"y": torch.tensor([[0.0], [1.0], [0.0]]), "w": torch.tensor([2.0, 0.5, 1.0])}
    loss = weighted_loss(nn.MSELoss())
    assert isinstance(loss, Criterion) and loss.sample_weight == "w"
    value = loss(outputs, batch, None)
    assert value.item() == pytest.approx(4.0 / 3.0) and value.requires_grad
    assert loss.function.reduction == "mean" and holder_with(loss, "reduction") is loss.function
    loss.function.reduction = "sum"
    assert loss(outputs, batch, None).item() == pytest.approx(4.0)
    loss.function.reduction = "none"
    with pytest.raises(RunError, match="mean or sum"):
        loss(outputs, batch, None)
    assert weighted_loss(nn.MSELoss(reduction="sum"))(outputs, batch, None).item() == pytest.approx(4.0)
    assert weighted_loss(functional.mse_loss)(outputs, batch, None).item() == pytest.approx(4.0 / 3.0)
    summed = weighted_loss(functools.partial(functional.l1_loss, reduction="sum"))
    assert summed(outputs, batch, None).item() == pytest.approx(2.0 + 1.0)
    plain = wrap_loss(nn.MSELoss(), {"pred": "net", "target": "y"}, "mse", "losses.mse")
    assert plain.sample_weight is None and plain.function.reduction == "mean"
    assert plain(outputs, batch, None).item() == pytest.approx(5.0 / 3.0)


def test_a_weight_spreads_over_the_other_dimensions():
    wide = {"net": torch.tensor([[1.0, 2.0], [0.0, 4.0]])}
    batch = {"y": torch.zeros(2, 2), "w": torch.tensor([3.0, 1.0])}
    assert weighted_loss(nn.MSELoss())(wide, batch, None).item() == pytest.approx((3 * 1 + 3 * 4 + 0 + 16) / 4)
    windows = {"y": torch.zeros(2, 2), "w": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
    assert weighted_loss(nn.L1Loss())(wide, windows, None).item() == pytest.approx((1 + 4 + 0 + 16) / 4)
    column = {"y": torch.zeros(2, 2), "w": torch.tensor([[3.0], [1.0]])}
    assert weighted_loss(nn.MSELoss())(wide, column, None).item() == pytest.approx((3 * 1 + 3 * 4 + 0 + 16) / 4)
    logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]])
    labels = {"y": torch.tensor([0, 2, 2]), "w": torch.tensor([[1.0], [5.0], [0.0]])}
    each = functional.cross_entropy(logits, labels["y"], reduction="none")
    wanted = (each * torch.tensor([1.0, 5.0, 0.0])).mean().item()
    assert weighted_loss(cross_entropy())({"net": logits}, labels, None).item() == pytest.approx(wanted)


def test_a_weighted_criterion_names_what_is_wrong():
    with pytest.raises(BuildError, match="reduction param.*log_cosh"):
        weighted_loss(log_cosh)
    with pytest.raises(BuildError, match="mean or sum"):
        weighted_loss(nn.MSELoss(reduction="none"))
    with pytest.raises(BuildError, match="names a batch field"):
        wrap_loss(nn.MSELoss(), {"sample_weight": 3}, "mse", "losses.mse")
    with pytest.raises(BuildError, match="sample_weight belongs to a criterion"):
        wrap_loss(Total(), {"sample_weight": "w"}, "total", "losses.total")
    loss = weighted_loss(nn.MSELoss())
    outputs = {"net": torch.zeros(3, 1)}
    with pytest.raises(RunError, match=r"loss 'mse'.*'w'.*\['y'\]"):
        loss(outputs, {"y": torch.zeros(3, 1)}, None)
    with pytest.raises(RunError, match=r"\(2,\).*\(3, 1\)"):
        loss(outputs, {"y": torch.zeros(3, 1), "w": torch.ones(2)}, None)
    with pytest.raises(RunError, match=r"\(3, 4\)"):
        loss(outputs, {"y": torch.zeros(3, 1), "w": torch.ones(3, 4)}, None)


def log_cosh(pred, target):
    return torch.log(torch.cosh(pred - target)).mean()


def config_of(path, **more):
    config = {
        "seed": 5,
        "device": "cpu",
        "data": {"use": "table", "file": str(path), **PARAMS, "weights": {"w": RULES}},
        "models": {"net": {"use": "mlp", "hidden": [8], "out": 1}},
        "losses": {"mse": {"use": "mse", "sample_weight": "w"}, "plain": "mse"},
        "optimizers": {"main": {"use": "adam", "lr": 0.01, "over": "@models.net", "minimizes": "@losses.mse"}},
        "train": {"epochs": 2},
    }
    for key, value in more.items():
        config[key] = {**config.get(key, {}), **value} if isinstance(value, dict) else value
    return config


def codes(checked):
    return [problem.code for problem in checked.problems if problem.severity == "error"]


def test_check_knows_the_weights(priced):
    path, frame = priced
    assert kalfa.check(config_of(path)).ok
    unknown = kalfa.check(config_of(path, losses={"mse": {"use": "mse", "sample_weight": "v"}}))
    assert codes(unknown) == ["unknown_weight"] and "the weights are w" in unknown.text()
    data = {key: value for key, value in config_of(path)["data"].items() if key != "weights"}
    missing = kalfa.check({**config_of(path), "data": data})
    assert codes(missing) == ["unknown_weight"] and "data.weights" in missing.text()
    assert "data_value" in codes(kalfa.check(config_of(path, data={"weights": {"x": RULES}})))
    total = {"use": "weighted_sum", "terms": [{"of": "@losses.plain"}], "sample_weight": "w"}
    assert "weight_objective" in codes(kalfa.check(config_of(path, losses={"total": total})))
    assert not kalfa.check(config_of(path, metrics={"rmse": {"use": "rmse", "sample_weight": "w"}})).ok
    assert any("sample_weight" in note and "w" in note for note in criterion_notes({"sample_weight": "w"}))


@pytest.mark.slow
def test_a_run_minimizes_the_weighted_loss(priced, tmp_path):
    path, frame = priced
    record = tmp_path / "runs" / "weighted"
    total = {"use": "weighted_sum", "terms": [{"of": "@losses.mse", "weight": 2.0}]}
    result = kalfa.run(config_of(path, losses={"total": total}, record=str(record)))
    history = read_lines(record / "history.jsonl")
    last = history[-1]
    assert [line["turn"] for line in history] == [1, 2]
    assert {"train/mse", "val/mse", "test/mse", "test/plain", "test/total"} <= set(last)
    context = result.context
    module = context.models["net"].module.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in context.data.loaders()["test"]:
            each = (module(batch["x"]) - batch["price"]).square().reshape(len(batch["x"]), -1).mean(dim=1)
            total += float((each * batch["w"]).sum())
            count += len(each)
    assert last["test/mse"] == pytest.approx(total / count, rel=1e-4)
    assert last["test/total"] == pytest.approx(2.0 * last["test/mse"], rel=1e-5)
    assert last["test/mse"] != pytest.approx(last["test/plain"], rel=1e-3)
    plan = json.loads((record / "plan.json").read_text())
    assert plan["nodes"]["losses.mse"]["keys"]["sample_weight"] == "w"
    fields = json.loads((record / "fitted" / "fields.json").read_text())
    assert fields["weights"][0]["name"] == "w" and fields["weights"][0]["scale"] > 0
    report = json.loads((record / "data.json").read_text())
    assert report["report"]["weights"]["w"]["scale"] == pytest.approx(fields["weights"][0]["scale"])
    assert any(node["kind"] == "weights" for node in report["pipeline"]["nodes"])
    predictions = pandas.read_parquet(record / "predictions.parquet")
    assert "w" not in predictions.columns and {"row", "price", "pred_net", "id"} <= set(predictions.columns)
    again = kalfa.predict(record)
    assert len(again.table) == len(predictions)
    fresh = tmp_path / "fresh.parquet"
    frame[["x1", "x2"]].head(9).to_parquet(fresh, index=False)
    assert len(kalfa.predict(record, data=str(fresh)).table) == 9
    assert Path(record / "predictions_fresh.parquet").is_file()
