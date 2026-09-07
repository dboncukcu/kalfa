"""The legos no reference config uses: the given split, weighted_sum, conv2d, maxpool, image_grid, recon_error."""

import functools
import warnings

import pandas
import pytest
import torch
from torch import nn

import kalfa  # noqa: F401
from helpers import batch, frame, tiny_model
from kalfa.std.adapter import criterion as criterion_adapter
from kalfa.std.criterion import mae, mse
from kalfa.std.layer import conv2d, maxpool
from kalfa.std.metric import recon_error
from kalfa.std.objective import weighted_sum
from kalfa.std.plot import image_grid, run_all
from kalfa.std.runtime import Context, entry_loss
from kalfa.std.split import given
from kalfa.synthetic import housing_frame, write_image_folder


def test_given_split_reads_the_other_sets_like_the_source(tmp_path):
    train = housing_frame(rows=20, seed=0)
    housing_frame(rows=6, seed=1).to_parquet(tmp_path / "valid.parquet", index=False)
    housing_frame(rows=4, seed=2).to_csv(tmp_path / "test.csv", index=False)
    parts = given(train, valid=str(tmp_path / "valid.parquet"), test=str(tmp_path / "test.csv"))
    assert len(parts["train"]) == 20 and len(parts["valid"]) == 6 and len(parts["test"]) == 4
    assert list(parts["test"].columns) == list(train.columns)
    parts = given(train)
    assert len(parts["valid"]) == 0 and len(parts["test"]) == 0 and list(parts["valid"].columns) == list(train.columns)
    with pytest.raises(ValueError, match="parquet or .csv"):
        given(train, valid=str(tmp_path / "valid.json"))
    from kalfa.std.source import image_folder

    write_image_folder(tmp_path / "a", classes=("x", "y"), per_class=3, size=8)
    write_image_folder(tmp_path / "b", classes=("x", "y"), per_class=2, size=8)
    parts = given(image_folder(str(tmp_path / "a")), test=str(tmp_path / "b"))
    assert len(parts["train"]) == 6 and len(parts["test"]) == 4 and len(parts["valid"]) == 0
    assert parts["test"].fields == ["image", "label"]


def test_weighted_sum_combines_other_losses_by_name():
    model = tiny_model(seed=1)
    losses = {"a": criterion_adapter(mse), "b": criterion_adapter(mae),
              "both": functools.partial(weighted_sum, terms={"a": 1.0, "b": 2.0})}
    context = Context(batch(), {"model": model}, predicts="model", targets=["price"], losses=losses,
                      losses_keys={"a": {}, "b": {}})
    value = entry_loss(losses["both"], context)
    a = float(entry_loss(losses["a"], context))
    b = float(entry_loss(losses["b"], context))
    assert set(value) == {"a", "b", "loss"}
    assert float(value["loss"]) == pytest.approx(a + 2.0 * b)
    assert value["loss"].requires_grad
    with pytest.raises(KeyError, match="no definition"):
        entry_loss(functools.partial(weighted_sum, terms={"ghost": 1.0}), context)
    with pytest.raises(ValueError, match="mapping"):
        weighted_sum(context.everything(), context.batch, terms=[], losses=None)


def test_conv2d_and_maxpool_shapes():
    net = nn.Sequential(conv2d(4, 3, padding=1), nn.ReLU(), maxpool(2), conv2d(2, 3, stride=2, padding=1, in_channels=4))
    out = net(torch.rand(2, 3, 16, 16))
    assert out.shape == (2, 2, 4, 4)
    assert isinstance(net[0], nn.LazyConv2d) or net[0].in_channels == 3


def test_recon_error_is_the_mean_per_sample_squared_error():
    metric = recon_error()
    metric.update(torch.tensor([[1.0, 1.0], [0.0, 0.0]]), torch.tensor([[0.0, 0.0], [0.0, 2.0]]))
    assert metric.compute() == pytest.approx((1.0 + 2.0) / 2)
    metric.reset()
    assert metric.compute() != metric.compute()


def test_image_grid_and_run_all_name_files_after_the_definition(tmp_path):
    from kalfa.std.feed import table
    from kalfa.std.loader import torch as torch_loader
    from kalfa.std.pre import apply, fit, to_tensor
    from kalfa.std.source import image_folder

    write_image_folder(tmp_path / "imgs", classes=("x",), per_class=4, size=8)
    samples = image_folder(str(tmp_path / "imgs"))
    prep = fit(samples, {"image": {"preprocessors": ["t"]}, "label": {"target": True}}, {"t": to_tensor()}, [])
    loader = torch_loader(table(apply(samples, prep, "test")), "test", {"size": 4})

    class Same(nn.Module):
        inputs = ["image"]
        outputs = ["out"]

        def forward(self, value):
            return value

    plots = {"grid": functools.partial(image_grid, n=3), "curve": functools.partial(image_grid, n=2, set="test")}
    run_all(None, [], {"same": Same()}, plots, keys={}, predicts="same", composites={}, valid_loader=None,
            test_loader=loader, record=str(tmp_path))
    assert (tmp_path / "plots" / "grid.png").exists() and (tmp_path / "plots" / "curve.png").exists()
    assert not (tmp_path / "plots" / "image_grid.png").exists()


def test_turn_warns_when_an_optimizer_takes_no_step():
    from kalfa.std.feed import table
    from kalfa.std.loader import torch as torch_loader
    from kalfa.std.optimizer import sgd
    from kalfa.std.turn import alternating

    first, second = tiny_model(seed=1), tiny_model(seed=2)
    optimizers = {"a": sgd({"first": first}, {"lr": 0.01}, None, "mse"),
                  "b": sgd({"second": second}, {"lr": 0.01}, None, "mse")}
    losses = {"mse": criterion_adapter(mse)}
    loader = torch_loader(table(frame(rows=8)), "train", {"size": 8})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = alternating({"first": first, "second": second}, optimizers, {}, {"global_step": 0, "turn": 0}, {}, {},
                          loader, {"order": ["a", "b"], "fresh_batch": True}, {}, losses, {}, {"mse": {}}, {},
                          "first", None, device="cpu")
    assert out["counters"]["global_step"] == 1
    assert any("['b'] took no step" in str(entry.message) for entry in caught)
