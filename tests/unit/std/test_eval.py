"""Evaluation of a set, the prediction table and generate."""

import math

import pandas
import pytest
import torch

import kalfa  # noqa: F401
from helpers import frame, tiny_model
from kalfa.std.adapter import criterion as criterion_adapter
from kalfa.std.adapter import metric as metric_adapter
from kalfa.std.criterion import mse
from kalfa.std.eval import evaluate, generate, predict, prediction_table
from kalfa.std.feed import table
from kalfa.std.loader import torch as torch_loader
from kalfa.std.metric import rmse
from kalfa.std.pre import Prep, apply, fit, standard_scaler
from kalfa.synthetic import housing_frame


def test_evaluate_empty_set_gives_an_empty_mapping():
    model = tiny_model()
    loader = torch_loader(table(frame(rows=0)), "valid", {"size": 4})
    assert evaluate({"model": model}, {}, {}, {"turn": 1}, {}, loader, "valid", {"l": criterion_adapter(mse)},
                    {"r": metric_adapter(rmse())}, {}, {}, "model") == {}


def test_evaluate_reports_losses_and_metrics_and_honours_every_and_sets():
    model = tiny_model()
    loader = torch_loader(table(frame(rows=12)), "valid", {"size": 5})
    losses = {"l": criterion_adapter(mse), "silent": criterion_adapter(mse)}
    metrics = {"r": metric_adapter(rmse()), "slow": metric_adapter(rmse())}
    losses_keys = {"silent": {"sets": ["test"]}}
    metrics_keys = {"slow": {"every": 2}, "r": {"target": "price"}}
    out = evaluate({"model": model}, {}, {}, {"turn": 1}, {}, loader, "valid", losses, metrics, losses_keys,
                   metrics_keys, "model")
    assert set(out) == {"l", "r"} and out["r"] == pytest.approx(math.sqrt(out["l"]), rel=1e-4)
    out = evaluate({"model": model}, {}, {}, {"turn": 2}, {}, loader, "valid", losses, metrics, losses_keys,
                   metrics_keys, "model")
    assert set(out) == {"l", "r", "slow"}
    assert not model.training


def test_prediction_table_inverts_the_target(tmp_path):
    data = housing_frame(rows=40)
    prep = fit(data, {"x*": {"preprocessors": ["s"]}, "price": {"target": True, "preprocessors": ["t"]}},
               {"s": standard_scaler(), "t": standard_scaler()}, [])
    test_frame = apply(data.iloc[30:], prep, "test")
    loader = torch_loader(table(test_frame), "test", {"size": 4})
    model = tiny_model(in_features=8)
    out = prediction_table(model, loader, prep, loader.dataset)
    assert list(out.columns) == ["row", "price", "raw_y", "pred_y"]
    assert out["row"].tolist() == list(range(30, 40))
    assert out["price"].to_numpy() == pytest.approx(data["price"].iloc[30:].to_numpy(), abs=1e-3)
    restored = prep.inverse("price", out["raw_y"].to_numpy())
    assert out["pred_y"].to_numpy() == pytest.approx(restored)
    written = predict({"model": model}, {}, loader, prep, "model", "test", record=str(tmp_path))
    assert (tmp_path / "predictions.parquet").exists() and len(written) == 10
    empty = predict({"model": model}, {}, torch_loader(table(frame(rows=0)), "test", {"size": 4}), prep, "model", "test")
    assert len(empty) == 0


def test_generate_without_a_section_does_nothing(tmp_path):
    assert generate({}, {}, None, None, str(tmp_path)) is None
    assert not (tmp_path / "samples").exists()
    called = {}

    def sampler(models, prep, rng):
        called["models"] = sorted(models)
        return torch.zeros(2)

    generate({"a": tiny_model()}, {"c": tiny_model()}, None, sampler, str(tmp_path))
    assert called["models"] == ["a", "c"] and (tmp_path / "samples" / "samples.pt").exists()
