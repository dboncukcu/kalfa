"""The legos no reference config uses: the given split, weighted_sum, mdmm, conv2d, maxpool, image_grid, recon_error."""

import functools
import warnings

import pytest
import torch
from torch import nn

import kalfa  # noqa: F401
from helpers import batch, frame, housing_frame, tiny_model, write_image_folder
from kalfa.std.adapter.kalfa.criterion import CriterionAdapter as criterion_adapter
from kalfa.std.adapter.kalfa.objective import ObjectiveAdapter
from kalfa.std.common.device import Device
from kalfa.std.common.runtime import Context, LossView, Pass
from kalfa.std.criterion.kalfa.regression import mae, mse
from kalfa.std.layer.kalfa.multipliers import Multipliers
from kalfa.std.layer.torch.convolution import conv2d
from kalfa.std.layer.torch.pooling import maxpool
from kalfa.std.lego.kalfa.run_all import run_all
from kalfa.std.metric.kalfa.recon_error import ReconError
from kalfa.std.objective.kalfa.mdmm import mdmm
from kalfa.std.objective.kalfa.weighted_sum import weighted_sum
from kalfa.std.plot.kalfa.images import image_grid
from kalfa.std.split.kalfa.splits import given


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
    from kalfa.std.source.kalfa.samples import image_folder

    write_image_folder(tmp_path / "a", classes=("x", "y"), per_class=3, size=8)
    write_image_folder(tmp_path / "b", classes=("x", "y"), per_class=2, size=8)
    parts = given(image_folder(str(tmp_path / "a")), test=str(tmp_path / "b"))
    assert len(parts["train"]) == 6 and len(parts["test"]) == 4 and len(parts["valid"]) == 0
    assert parts["test"].fields == ["image", "label"]


def test_weighted_sum_combines_other_losses_by_name():
    model = tiny_model(seed=1)
    losses = {"a": criterion_adapter(mse), "b": criterion_adapter(mae),
              "both": ObjectiveAdapter(functools.partial(weighted_sum, terms={"a": 1.0, "b": 2.0}))}
    context = Context(batch(), Pass({"model": model}, predicts="model", targets=["price"], losses=losses,
                                    losses_keys={"a": {}, "b": {}}))
    value = losses["both"].loss(context)
    a = float(losses["a"].loss(context).detach())
    b = float(losses["b"].loss(context).detach())
    assert set(value) == {"a", "b", "loss"}
    assert float(value["loss"].detach()) == pytest.approx(a + 2.0 * b)
    assert value["loss"].requires_grad
    with pytest.raises(KeyError, match="no definition"):
        ObjectiveAdapter(functools.partial(weighted_sum, terms={"ghost": 1.0})).loss(context)
    with pytest.raises(ValueError, match="mapping"):
        weighted_sum(context.scope.everything(), context.batch, terms=[], losses=None)


def multipliers_model(constraints, name="lambdas"):
    from cirak.build import Graph, GraphNode

    from kalfa.std.builder.kalfa.module import Module

    layer = Multipliers(constraints)
    return Module(Graph(("x",), ("lmbda",), (GraphNode("lmbda", layer, ("x",), ("lmbda",)),)), name=name)


def test_mdmm_holds_a_term_at_its_epsilon_through_a_multiplier():
    model = tiny_model(seed=1)
    constraints = {"b": {"epsilon": 0.5, "lmbda_init": -1.0, "scale": 2.0, "damping": 0.5}, "w.a": {"epsilon": 0.1}}
    lambdas = multipliers_model(constraints)
    losses = {"a": criterion_adapter(mse), "b": criterion_adapter(mae),
              "w": ObjectiveAdapter(functools.partial(weighted_sum, terms={"a": 1.0, "b": 1.0})),
              "total": ObjectiveAdapter(functools.partial(mdmm, primary="a", multipliers="lambdas",
                                                          constraints=constraints))}
    context = Context(batch(), Pass({"model": model, "lambdas": lambdas}, predicts="model", targets=["price"],
                                    losses=losses, losses_keys={"a": {}, "b": {}}))
    value = losses["total"].loss(context)
    a = float(losses["a"].loss(context).detach())
    b = float(losses["b"].loss(context).detach())
    assert set(value) == {"loss", "primary", "lambda/b", "inf/b", "lambda/w.a", "inf/w.a"}
    inf_b, inf_a = 0.5 - b, 0.1 - a
    assert float(value["primary"].detach()) == pytest.approx(a)
    assert float(value["inf/b"].detach()) == pytest.approx(inf_b)
    assert float(value["inf/w.a"].detach()) == pytest.approx(inf_a)
    assert float(value["lambda/b"].detach()) == -1.0 and float(value["lambda/w.a"].detach()) == 0.0
    expected = a + 2.0 * (-1.0 * inf_b + 0.5 * inf_b ** 2 / 2) + (0.0 * inf_a + inf_a ** 2 / 2)
    assert float(value["loss"].detach()) == pytest.approx(expected, rel=1e-5)
    value["loss"].backward()
    lmbda = next(lambdas.parameters())
    assert lmbda.grad.tolist() == pytest.approx([2.0 * inf_b, inf_a], rel=1e-5)
    assert model.nodes["layer"].weight.grad is not None
    tracked = losses["total"].tracker("total")
    tracked.observe(context)
    assert set(tracked.result()) == {"total", "total/primary", "total/lambda/b", "total/inf/b", "total/lambda/w.a",
                                     "total/inf/w.a"}
    view = LossView(context)
    with pytest.raises(KeyError, match="no definition"):
        mdmm(context.scope.everything(), context.batch, view, "ghost", "lambdas", constraints)
    with pytest.raises(KeyError, match="no term"):
        mdmm({"other": multipliers_model({"w.ghost": 0.0})}, context.batch, view, "a", "other", {"w.ghost": 0.0})
    with pytest.raises(KeyError, match="no model"):
        mdmm(context.scope.everything(), context.batch, view, "a", "ghost", constraints)
    with pytest.raises(ValueError, match="stay in step"):
        mdmm(context.scope.everything(), context.batch, view, "a", "lambdas", {"w.a": 0.1, "b": 0.5})
    with pytest.raises(ValueError, match="epsilon"):
        mdmm({}, {}, view, "a", "lambdas", {"b": {"lmbda_init": 1.0}})
    with pytest.raises(ValueError, match="mapping"):
        mdmm({}, {}, view, "a", "lambdas", [])
    with pytest.raises(ValueError, match="has no"):
        mdmm({}, {}, view, "a", "lambdas", {"b": {"epsilon": 0.0, "weight": 1.0}})


def test_conv2d_and_maxpool_shapes():
    net = nn.Sequential(conv2d(4, 3, padding=1), nn.ReLU(), maxpool(2),
                        conv2d(2, 3, stride=2, padding=1, in_channels=4))
    out = net(torch.rand(2, 3, 16, 16))
    assert out.shape == (2, 2, 4, 4)
    assert isinstance(net[0], nn.LazyConv2d) or net[0].in_channels == 3


def test_recon_error_is_the_mean_per_sample_squared_error():
    metric = ReconError()
    metric.update(torch.tensor([[1.0, 1.0], [0.0, 0.0]]), torch.tensor([[0.0, 0.0], [0.0, 2.0]]))
    assert metric.compute() == pytest.approx((1.0 + 2.0) / 2)
    metric.reset()
    assert metric.compute() != metric.compute()


def test_image_grid_and_run_all_name_files_after_the_definition(tmp_path):
    from kalfa.std.feed.kalfa.table import table
    from kalfa.std.lego.kalfa.prep import apply, fit
    from kalfa.std.loader.kalfa.torch import torch_loader
    from kalfa.std.pre.kalfa.images import ToTensor
    from kalfa.std.source.kalfa.samples import image_folder

    write_image_folder(tmp_path / "imgs", classes=("x",), per_class=4, size=8)
    samples = image_folder(str(tmp_path / "imgs"))
    prep = fit(samples, {"image": {"preprocessors": ["t"]}, "label": {"target": True}}, {"t": ToTensor()}, [])
    loader = torch_loader(table(apply(samples, prep, "test")), "test", 4)

    class Same(nn.Module):
        inputs = ["image"]
        outputs = ["out"]

        def forward(self, value):
            return value

    plots = {"grid": functools.partial(image_grid, n=3), "curve": functools.partial(image_grid, n=2, set="test")}
    run_all(None, [], {"same": Same()}, plots, keys={}, predicts="same",
            bus={"composites": {}, "valid_loader": None, "test_loader": loader}, record=str(tmp_path))
    assert (tmp_path / "plots" / "grid.png").exists() and (tmp_path / "plots" / "curve.png").exists()
    assert not (tmp_path / "plots" / "image_grid.png").exists()


def test_turn_warns_when_an_optimizer_takes_no_step():
    from kalfa.std.feed.kalfa.table import table
    from kalfa.std.loader.kalfa.torch import torch_loader
    from kalfa.std.optimizer.torch.optimizers import Sgd
    from kalfa.std.turn.kalfa.alternating import alternating

    first, second = tiny_model(seed=1), tiny_model(seed=2)
    optimizers = {"a": Sgd({"first": first}, {"lr": 0.01}, None, "mse"),
                  "b": Sgd({"second": second}, {"lr": 0.01}, None, "mse")}
    losses = {"mse": criterion_adapter(mse)}
    loader = torch_loader(table(frame(rows=8)), "train", 8)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = alternating({"first": first, "second": second}, optimizers, {}, {"global_step": 0, "turn": 0}, {}, {},
                          loader, {"order": ["a", "b"], "fresh_batch": True}, {}, losses, {}, {"mse": {}}, {},
                          "first", None, device=Device.cpu())
    assert out["counters"]["global_step"] == 1
    assert any("['b'] took no step" in str(entry.message) for entry in caught)
