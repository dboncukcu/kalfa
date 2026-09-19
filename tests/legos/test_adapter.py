import functools
import math

import numpy
import pytest
import torch
from cirak.build import Deferred
from cirak.registry import registry

from helpers import batch, build, frame, tiny_model
from kalfa.std import STD_URIS
from kalfa.std.adapter.kalfa.criterion import CriterionAdapter, MeanTracker
from kalfa.std.adapter.kalfa.metric import MetricAdapter, MetricTracker
from kalfa.std.adapter.kalfa.objective import ObjectiveAdapter, ObjectiveTracker
from kalfa.std.common.runtime import Context, LossView, Pass
from kalfa.std.pre.base import Field, Grouped, Prep


ADAPTERS = sorted(uri for uri in STD_URIS if uri.startswith("/adapter/"))


class Probe:
    def __init__(self):
        self.calls = []

    def reset(self):
        self.calls = []

    def update(self, models, batch, rng, predicts, record, turn, prep, set):
        self.calls.append({"models": models, "batch": batch, "rng": rng, "predicts": predicts, "record": record,
                           "turn": turn, "prep": prep, "set": set})

    def compute(self):
        return len(self.calls)


def constant(model, weight, bias=0.0):
    with torch.no_grad():
        model.nodes["layer"].weight.fill_(weight)
        model.nodes["layer"].bias.fill_(bias)
    return model


def scope(model, targets=("price",), **extra):
    return Pass(models={"m": model}, predicts="m", targets=list(targets), **extra)


def summed_model():
    return constant(tiny_model(), 1.0)


def price_prep(scaler):
    fields = [Field("x0", [], False, ["x0"]), Field("x1", [], False, ["x1"]), Field("x2", [], False, ["x2"]),
              Field("price", ["scale"], True, ["price"])]
    return Prep(fields, {"scale": {"price": scaler}}, {}, {}, [])


def criterion(uri, **params):
    return build("/adapter/kalfa/criterion", criterion=build(uri, **params))


def train_loader():
    dataset = build("/feed/kalfa/table", frame=frame(rows=8))
    return build("/loader/kalfa/torch", data=dataset, set="train", size=4, shuffle=False)


def test_adapter_scope_is_the_three_kalfa_adapters():
    assert ADAPTERS == ["/adapter/kalfa/criterion", "/adapter/kalfa/metric", "/adapter/kalfa/objective"]


def test_prediction_adapters_declare_that_they_use_the_predicts_model():
    assert registry.facts("/adapter/kalfa/criterion").get("uses") == ["predicts"]
    assert registry.facts("/adapter/kalfa/metric").get("uses") == ["predicts"]
    assert registry.facts("/adapter/kalfa/objective").get("uses") is None


def test_adapters_say_what_they_read():
    assert build("/adapter/kalfa/criterion", criterion=build("/criterion/kalfa/mse")).reads == "predictions"
    assert build("/adapter/kalfa/metric", metric=build("/metric/kalfa/rmse")).reads == "predictions"
    objective = build("/objective/kalfa/weighted_sum", terms={"a": 1.0})
    assert build("/adapter/kalfa/objective", objective=objective).reads == "models"


def test_criterion_adapter_feeds_the_predicts_output_and_the_single_target():
    adapter = criterion("/criterion/kalfa/mse")
    assert isinstance(adapter, CriterionAdapter)
    data = batch()
    context = Context(data, scope(summed_model()))
    value = adapter.loss(context)
    expected = float(((data["x"].sum(dim=1) - data["price"]) ** 2).mean())
    assert float(value.detach()) == pytest.approx(expected)
    assert value.requires_grad


def test_criterion_adapter_reads_the_output_and_target_keys_of_the_definition():
    adapter = criterion("/criterion/kalfa/mae")
    data = batch()
    data["size"] = torch.ones(8)
    context = Context(data, scope(summed_model(), targets=("price", "size")))
    value = adapter.loss(context, {"output": "y", "target": "size"})
    assert float(value.detach()) == pytest.approx(float((data["x"].sum(dim=1) - 1.0).abs().mean()))


def test_criterion_adapter_target_glob_and_list_select_the_fields():
    adapter = criterion("/criterion/kalfa/mse")
    data = batch()
    data["size"] = torch.ones(8)
    context = Context(data, scope(constant(tiny_model(3, 2), 0.0), targets=("price", "size")))
    assert context.target_fields("pr*") == ["price"]
    assert context.target_fields(["price", "size"]) == ["price", "size"]
    stacked = context.target(["price", "size"])
    assert stacked.shape == (8, 2)
    assert torch.equal(stacked[:, 1], torch.ones(8))
    value = adapter.loss(context, {"target": ["price", "size"]})
    assert float(value.detach()) == pytest.approx(float((stacked ** 2).mean()))


def test_criterion_adapter_reads_the_targets_table_for_the_output_wire():
    adapter = criterion("/criterion/kalfa/mae")
    data = batch()
    data["size"] = torch.ones(8)
    context = Context(data, scope(summed_model(), targets=("price", "size"), target_map={"y": "size"}))
    value = adapter.loss(context, {"output": "y"})
    assert float(value.detach()) == pytest.approx(float((data["x"].sum(dim=1) - 1.0).abs().mean()))
    lone = Context(data, scope(summed_model(), targets=("price", "size"), target_map={"y": "size"}))
    assert lone.selector() == "size"


def test_criterion_adapter_target_input_reads_the_model_input():
    adapter = criterion("/criterion/kalfa/mse")
    data = batch()
    context = Context(data, scope(constant(tiny_model(3, 3), 0.0)))
    value = adapter.loss(context, {"target": "input"})
    assert float(value.detach()) == pytest.approx(float((data["x"] ** 2).mean()))


def test_context_refuses_an_unwritten_target_among_several_fields():
    data = batch()
    data["size"] = torch.ones(8)
    context = Context(data, scope(summed_model(), targets=("price", "size")))
    with pytest.raises(ValueError, match=r"target is not written and the batch has 2 target fields "
                                          r"\['price', 'size'\]; write target on the definition, or training.targets"):
        criterion("/criterion/kalfa/mse").loss(context)


def test_context_names_an_unknown_output_wire():
    context = Context(batch(), scope(summed_model()))
    with pytest.raises(KeyError, match=r"model 'm' has no output wire 'z'; it writes \['y'\]"):
        criterion("/criterion/kalfa/mse").loss(context, {"output": "z"})


def test_context_names_a_target_glob_that_matches_no_field():
    context = Context(batch(), scope(summed_model()))
    with pytest.raises(KeyError, match=r"target 'si\*' names no target field; the fields are \['price'\]"):
        criterion("/criterion/kalfa/mse").loss(context, {"target": "si*"})
    with pytest.raises(KeyError, match=r"target fields \['size'\] are not in the batch; the fields are "
                                        r"\['price', 'x'\]"):
        criterion("/criterion/kalfa/mse").loss(context, {"target": "size"})


def test_context_names_the_target_fields_missing_from_the_batch():
    context = Context({"x": batch()["x"]}, scope(summed_model()))
    with pytest.raises(KeyError, match=r"target fields \['price'\] are not in the batch; the fields are \['x'\]"):
        criterion("/criterion/kalfa/mse").loss(context)


def test_context_caches_the_outputs_and_detaches_a_copy():
    context = Context(batch(), scope(summed_model()))
    assert context.computed is None
    first = context.outputs()
    assert list(first) == ["y"]
    assert context.outputs() is first
    assert context.predictions() is first["y"]
    assert context.size == 8
    copy = context.detached()
    assert copy.computed["y"].requires_grad is False
    assert torch.equal(copy.computed["y"], first["y"].detach())
    with pytest.raises(ValueError, match=r"a batch needs at least one tensor field"):
        Context({"name": "x"}, scope(summed_model())).size


def test_pass_resolves_models_composites_and_ema_copies_by_name():
    model = summed_model()
    other = tiny_model(index=1)
    ema = tiny_model(index=2)
    scoped = Pass(models={"m": model}, composites={"full": other}, emas={"m": ema}, predicts="m", targets=["price"])
    assert scoped.model() is model
    assert scoped.everything() == {"m": model, "full": other, "m.ema": ema}
    scoped.predicts = "full"
    assert scoped.model() is other
    scoped.predicts = "m.ema"
    assert scoped.model() is ema
    scoped.predicts = "full.ema"
    with pytest.raises(KeyError, match=r"model 'full' has no ema copy"):
        scoped.model()
    scoped.predicts = "q"
    with pytest.raises(KeyError, match=r"unknown model 'q'; the models are \['full', 'm'\]"):
        scoped.model()
    scoped.predicts = None
    with pytest.raises(ValueError, match=r"no model name given; write training.predicts"):
        scoped.model()


def test_rescale_inverts_an_affine_scaler_on_the_device():
    scaler = build("/pre/sklearn/standard_scaler")
    scaler.fit(numpy.array([1.0, 3.0, 5.0, 7.0]))
    prep = price_prep(scaler)
    assert prep.rescales("price") and prep.rescales_on_device(["price"], "test")
    data = batch()
    context = Context(data, scope(summed_model(), prep=prep, set_name="test"))
    predictions, targets = context.rescaled("y", "price")
    mean, scale = 4.0, math.sqrt(5.0)
    assert torch.allclose(predictions, (data["x"].sum(dim=1, keepdim=True) * scale + mean).float())
    assert torch.allclose(targets, (data["price"] * scale + mean).float())
    again = context.rescaled("y", "price")
    assert again[0] is predictions and again[1] is targets
    adapter = criterion("/criterion/kalfa/mse")
    value = adapter.loss(context, {"output": "y", "target": "price"}, rescale=True)
    expected = float(((predictions.detach().reshape(-1) - targets) ** 2).mean())
    assert float(value) == pytest.approx(expected)
    assert value.requires_grad is False


def test_rescale_inverts_a_numpy_only_scaler_through_its_columns():
    scaler = build("/pre/kalfa/tanh", scale=2.0)
    prep = price_prep(scaler)
    assert prep.rescales_on_device(["price"], "test") is False
    data = batch()
    data["price"] = torch.tanh(data["price"])
    context = Context(data, scope(constant(tiny_model(), 0.1), prep=prep, set_name="test"))
    predictions, targets = context.rescaled("y", "price")
    guess = (data["x"].sum(dim=1, keepdim=True) * 0.1).double().numpy()
    assert numpy.allclose(predictions.numpy(), numpy.arctanh(guess) * 2.0, atol=1e-6)
    assert numpy.allclose(targets.numpy(), numpy.arctanh(data["price"].double().numpy()) * 2.0, atol=1e-6)


def test_rescale_without_a_prep_or_a_scaler_returns_the_values_as_they_are():
    data = batch()
    context = Context(data, scope(summed_model()))
    predictions, targets = context.rescaled("y", "price")
    assert predictions is context.predictions("y")
    assert targets is data["price"]
    plain = Prep([Field("price", [], True, ["price"])], {}, {}, {}, [])
    scoped = Context(data, scope(summed_model(), prep=plain))
    assert scoped.rescaled("y", "price")[1] is data["price"]


def test_rescale_of_target_input_inverts_the_feature_columns():
    scaler = build("/pre/sklearn/standard_scaler")
    scaler.fit(numpy.array([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]]))
    fields = [Field("x0", ["scale"], False, ["x0"]), Field("x1", ["scale"], False, ["x1"]),
              Field("x2", ["scale"], False, ["x2"])]
    prep = Prep(fields, {"scale": Grouped(scaler, ["x0", "x1", "x2"])}, {}, {}, [])
    data = batch()
    context = Context(data, scope(constant(tiny_model(3, 3), 0.0), prep=prep, set_name="test"))
    predictions, targets = context.rescaled("y", "input")
    assert torch.allclose(predictions, torch.tensor([[1.0, 2.0, 3.0]]).expand(8, 3))
    assert torch.allclose(targets, data["x"] * torch.tensor([1.0, 2.0, 3.0]) + torch.tensor([1.0, 2.0, 3.0]),
                          atol=1e-5)


def test_mean_tracker_averages_the_loss_over_the_observed_rows():
    adapter = criterion("/criterion/kalfa/mse")
    tracker = adapter.tracker("mse", {"output": "y"})
    assert isinstance(tracker, MeanTracker)
    assert math.isnan(tracker.result()["mse"])
    first, second = batch(rows=8, seed=1), batch(rows=2, seed=2)
    tracker.observe(Context(first, scope(summed_model())))
    tracker.observe(Context(second, scope(summed_model())))
    loss_of = build("/criterion/kalfa/mse")
    total = 8 * float(loss_of(first["x"].sum(dim=1), first["price"])) \
        + 2 * float(loss_of(second["x"].sum(dim=1), second["price"]))
    assert tracker.result() == {"mse": pytest.approx(total / 10)}
    tracker.record(torch.tensor(1.0), 10)
    assert tracker.result() == {"mse": pytest.approx((total + 10.0) / 20)}


def test_metric_tracker_runs_a_copy_of_the_metric_over_the_pass():
    adapter = build("/adapter/kalfa/metric", metric=build("/metric/kalfa/rmse"))
    assert isinstance(adapter, MetricAdapter)
    tracker = adapter.tracker("rmse", {"output": "y", "target": "price"}, rescale=True)
    assert isinstance(tracker, MetricTracker)
    assert tracker.live is not adapter.metric
    assert math.isnan(tracker.result()["rmse"])
    data = batch()
    tracker.observe(Context(data, scope(summed_model())))
    expected = math.sqrt(float(((data["x"].sum(dim=1) - data["price"]) ** 2).mean()))
    assert tracker.result() == {"rmse": pytest.approx(expected)}
    assert tracker.seen == 8
    assert adapter.metric.count == 0


def test_metric_tracker_hands_the_run_context_to_a_metric_that_names_it():
    probe = Probe()
    adapter = build("/adapter/kalfa/metric", metric=probe)
    ema = tiny_model(index=3)
    rng = torch.Generator()
    scoped = Pass(models={"m": summed_model()}, emas={"m": ema}, predicts="m", targets=["price"], epoch=4, rng=rng,
                  set_name="valid", record="rec")
    tracker = adapter.tracker("probe")
    data = batch()
    tracker.observe(Context(data, scoped))
    call = tracker.live.calls[0]
    assert call["models"] == {"m": scoped.models["m"], "m.ema": ema}
    assert call["batch"] is data
    assert call["rng"] is rng
    assert call["predicts"] == "m"
    assert call["record"] == "rec"
    assert call["turn"] == 4
    assert call["prep"] is None
    assert call["set"] == "valid"
    assert tracker.result() == {"probe": 1.0}
    assert probe.calls == []


def test_metric_tracker_reports_nothing_for_a_metric_that_computes_none():
    adapter = build("/adapter/kalfa/metric", metric=build("/metric/kalfa/sample_writer", n=2))
    tracker = adapter.tracker("writer")
    tracker.observe(Context(batch(), scope(summed_model())))
    assert tracker.result() == {}


def test_metric_adapter_is_observed_never_minimized_and_takes_no_effect():
    adapter = build("/adapter/kalfa/metric", metric=build("/metric/kalfa/rmse"))
    with pytest.raises(ValueError, match=r"a metric is observed, never minimized; write it under metrics"):
        adapter.loss(Context(batch(), scope(summed_model())))
    with pytest.raises(ValueError, match=r"a metric takes no rule effect; 'delta' cannot be set on it"):
        adapter.with_param("delta", 2.0)
    tracker = adapter.tracker("rmse")
    with pytest.raises(ValueError, match=r"metric 'rmse' observes batches; it takes no recorded value"):
        tracker.record(1.0, 4)


def test_objective_adapter_calls_the_objective_with_every_model_and_the_losses_view():
    objective = build("/objective/kalfa/weighted_sum", terms={"a": 1.0, "b": 0.5})
    adapter = build("/adapter/kalfa/objective", objective=objective)
    assert isinstance(adapter, ObjectiveAdapter)
    data = batch()
    losses = {"a": criterion("/criterion/kalfa/mse"), "b": criterion("/criterion/kalfa/mae")}
    context = Context(data, scope(summed_model(), losses=losses, losses_keys={"a": {"output": "y"}}))
    out = adapter.loss(context)
    guess = data["x"].sum(dim=1)
    mse = float(((guess - data["price"]) ** 2).mean())
    mae = float((guess - data["price"]).abs().mean())
    assert list(out) == ["a", "b", "loss"]
    assert float(out["loss"].detach()) == pytest.approx(mse + 0.5 * mae)
    assert out["loss"].requires_grad


def test_objective_adapter_passes_only_the_arguments_the_signature_names():
    seen = {}

    def probe(models, batch, step=None, epoch=None, rng=None, scaler=None, losses=None):
        seen.update({"models": models, "batch": batch, "step": step, "epoch": epoch, "rng": rng, "scaler": scaler,
                     "losses": losses})
        return torch.tensor(1.0)

    def plain(models, batch):
        seen["plain"] = (models, batch)
        return torch.tensor(2.0)

    rng = torch.Generator()
    other = tiny_model(index=1)
    scoped = Pass(models={"m": summed_model()}, composites={"full": other}, predicts="m", targets=["price"],
                  epoch=3, rng=rng)
    data = batch()
    context = Context(data, scoped, step=17)
    assert float(build("/adapter/kalfa/objective", objective=probe).loss(context)) == 1.0
    assert seen["models"] == {"m": scoped.models["m"], "full": other}
    assert seen["batch"] is data
    assert seen["step"] == 17
    assert seen["epoch"] == 3
    assert seen["rng"] is rng
    assert seen["scaler"] is None
    assert isinstance(seen["losses"], LossView)
    assert float(build("/adapter/kalfa/objective", objective=plain).loss(context)) == 2.0
    assert seen["plain"] == ({"m": scoped.models["m"], "full": other}, data)


def test_objective_tracker_names_every_term_under_the_definition():
    objective = build("/objective/kalfa/weighted_sum", terms={"a": 1.0, "b": 0.5})
    adapter = build("/adapter/kalfa/objective", objective=objective)
    tracker = adapter.tracker("ws")
    assert isinstance(tracker, ObjectiveTracker)
    assert math.isnan(tracker.result()["ws"])
    losses = {"a": criterion("/criterion/kalfa/mse"), "b": criterion("/criterion/kalfa/mae")}
    data = batch()
    tracker.observe(Context(data, scope(summed_model(), losses=losses)))
    guess = data["x"].sum(dim=1)
    mse = float(((guess - data["price"]) ** 2).mean())
    mae = float((guess - data["price"]).abs().mean())
    assert tracker.result() == {"ws/a": pytest.approx(mse), "ws/b": pytest.approx(mae),
                                "ws": pytest.approx(mse + 0.5 * mae)}
    tracker.record({"loss": torch.tensor(4.0), "a": 2.0, "b": 0.0}, 8)
    assert tracker.result() == {"ws/a": pytest.approx((mse + 2.0) / 2), "ws/b": pytest.approx(mae / 2),
                                "ws": pytest.approx((mse + 0.5 * mae + 4.0) / 2)}


def test_objective_tracker_records_a_bare_value_as_the_loss():
    adapter = build("/adapter/kalfa/objective", objective=build("/objective/kalfa/weighted_sum", terms={"a": 1.0}))
    tracker = adapter.tracker("ws")
    tracker.record(torch.tensor(3.0), 2)
    tracker.record(1.0, 2)
    assert tracker.result() == {"ws": 2.0}


def test_loss_view_reads_the_losses_table_with_the_definition_keys():
    losses = {"a": criterion("/criterion/kalfa/mse"), "b": criterion("/criterion/kalfa/mae")}
    data = batch()
    data["size"] = torch.ones(8)
    context = Context(data, scope(summed_model(), targets=("price", "size"), losses=losses,
                                  losses_keys={"a": {"target": "price"}, "b": {"target": "size"}}))
    view = LossView(context)
    assert "a" in view and "zz" not in view
    guess = data["x"].sum(dim=1)
    assert float(view["a"].detach()) == pytest.approx(float(((guess - data["price"]) ** 2).mean()))
    assert float(view["b"].detach()) == pytest.approx(float((guess - 1.0).abs().mean()))
    with pytest.raises(KeyError, match=r"losses has no definition 'zz'; the definitions are \['a', 'b'\]"):
        view["zz"]


def test_with_param_rebinds_a_criterion_param_on_a_copy():
    adapter = criterion("/criterion/kalfa/huber", delta=1.0)
    changed = adapter.with_param("delta", 3.0)
    assert isinstance(changed, CriterionAdapter) and changed is not adapter
    assert changed.criterion.keywords == {"delta": 3.0}
    assert adapter.criterion.keywords == {"delta": 1.0}
    context = Context(batch(), scope(summed_model()))
    wide = build("/criterion/kalfa/huber", delta=3.0)
    expected = float(wide(context.predictions(), context.target()).detach())
    assert float(changed.loss(context).detach()) == pytest.approx(expected)


def test_with_param_wraps_a_bare_criterion_in_a_partial():
    adapter = criterion("/criterion/kalfa/huber")
    changed = adapter.with_param("delta", 2.0)
    assert isinstance(changed.criterion, functools.partial)
    assert changed.criterion.keywords == {"delta": 2.0}
    assert changed.criterion.func is adapter.criterion


def test_with_param_sets_a_nested_objective_param_without_touching_the_original():
    terms = {"a": 1.0, "b": 0.5}
    adapter = build("/adapter/kalfa/objective", objective=build("/objective/kalfa/weighted_sum", terms=terms))
    changed = adapter.with_param("terms.a", 2.0)
    assert changed.objective.keywords == {"terms": {"a": 2.0, "b": 0.5}}
    assert adapter.objective.keywords == {"terms": {"a": 1.0, "b": 0.5}}
    assert terms == {"a": 1.0, "b": 0.5}
    replaced = adapter.with_param("terms", {"a": 3.0})
    assert replaced.objective.keywords == {"terms": {"a": 3.0}}
    deep = build("/adapter/kalfa/objective", objective=build(
        "/objective/kalfa/mdmm", primary="ws", multipliers="lambdas", constraints={"mae": {"epsilon": 0.5}}))
    assert deep.with_param("constraints.mae.scale", 2.0).objective.keywords["constraints"] == {
        "mae": {"epsilon": 0.5, "scale": 2.0}}


def test_resolve_builds_a_deferred_criterion_param_with_the_loader():
    deferred = Deferred("/data/kalfa/target_weights", {"weights": {"price": 3.0, "default": 1.0}},
                        registry.resolve("/data/kalfa/target_weights"))
    adapter = criterion("/criterion/kalfa/weighted_mse", weights=deferred)
    assert adapter.resolve(loader=train_loader()) is adapter
    assert torch.equal(adapter.criterion.keywords["weights"], torch.tensor([3.0]))
    data = batch()
    context = Context(data, scope(summed_model()))
    expected = 3.0 * float(((data["x"].sum(dim=1) - data["price"]) ** 2).mean())
    assert float(adapter.loss(context).detach()) == pytest.approx(expected)


def test_resolve_leaves_a_criterion_without_deferred_params_alone():
    adapter = criterion("/criterion/kalfa/huber", delta=1.0)
    before = adapter.criterion
    adapter.resolve(loader=train_loader())
    assert adapter.criterion is before
    bare = criterion("/criterion/kalfa/mse")
    function = bare.criterion
    bare.resolve(loader=None)
    assert bare.criterion is function


def test_resolve_builds_a_deferred_objective_param():
    def probe(models, batch, weights):
        return weights

    deferred = Deferred("/data/kalfa/target_weights", {"weights": {"default": 2.0}},
                        registry.resolve("/data/kalfa/target_weights"))
    adapter = build("/adapter/kalfa/objective", objective=functools.partial(probe, weights=deferred))
    adapter.resolve(loader=train_loader())
    assert torch.equal(adapter.loss(Context(batch(), scope(summed_model()))), torch.tensor([2.0]))
