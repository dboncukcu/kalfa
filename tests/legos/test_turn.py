import copy
import functools
import json
import math
import re

import pytest
import torch
from cirak.registry import registry

from helpers import build, frame, tiny_model
from kalfa.std import STD_URIS
from kalfa.std.turn.base import Cursor


TURNS = sorted(uri for uri in STD_URIS if uri.startswith("/turn/"))


class Monitor:
    def __init__(self):
        self.begun = []
        self.lines = []

    def turn_begins(self, count):
        self.begun.append(count)

    def step(self, line):
        self.lines.append(dict(line))


def loader_of(rows=16, size=4, seed=0):
    dataset = build("/feed/kalfa/table", frame=frame(rows=rows, seed=seed))
    return build("/loader/kalfa/torch", data=dataset, set="train", size=size, shuffle=False)


def criterion(uri, **params):
    return build("/adapter/kalfa/criterion", criterion=build(uri, **params))


def own_loss(models, batch, name, trace=None):
    if trace is not None:
        trace.append((name, batch["x"]))
    guess = models[name](batch["x"]).reshape(-1)
    return ((guess - batch["price"]) ** 2).mean()


def objective(name, trace=None):
    return build("/adapter/kalfa/objective", objective=functools.partial(own_loss, name=name, trace=trace))


def sgd(name, model, loss, lr=0.1):
    return build("/optimizer/torch/sgd", models={name: model}, params={"lr": lr}, loss=loss)


def turn(models, optimizers, loader, losses, params=None, extra=None, metrics=None, losses_keys=None,
         metrics_keys=None, predicts="m", steps=None, stream=None, record=None, emas=None, counters=None,
         effects=None, monitor=None):
    return build("/turn/kalfa/alternating", models=models, optimizers=optimizers, emas=emas or {},
                 counters=counters if counters is not None else {}, composites={}, effects=effects or {},
                 loader=loader, params=params or {}, extra=extra or {}, losses=losses, metrics=metrics or {},
                 losses_keys=losses_keys or {}, metrics_keys=metrics_keys or {}, predicts=predicts, steps=steps,
                 stream=stream, record=record, monitor=monitor)


def replay(model, batches, loss_of, lr=0.1, accumulate=1, grad_clip=None, ema=None, decay=0.5):
    twin = copy.deepcopy(model)
    shadow = copy.deepcopy(ema)
    optimizer = torch.optim.SGD(twin.parameters(), lr=lr)
    losses, norms = [], []
    for start in range(0, len(batches), accumulate):
        group = batches[start:start + accumulate]
        optimizer.zero_grad()
        for data in group:
            loss = loss_of(twin(data["x"]).reshape(-1), data["price"])
            (loss / len(group)).backward()
            losses.append((float(loss.detach()), len(data["x"])))
        if grad_clip is not None:
            norms.append(float(torch.nn.utils.clip_grad_norm_(twin.parameters(), grad_clip)))
        optimizer.step()
        if shadow is not None:
            with torch.no_grad():
                for own, theirs in zip(shadow.parameters(), twin.parameters()):
                    own.mul_(decay).add_(theirs, alpha=1.0 - decay)
    mean = sum(value * size for value, size in losses) / sum(size for _, size in losses)
    return twin, mean, norms, shadow


def observed_mean(model, batches, trained_with, watched, lr=0.1):
    twin = copy.deepcopy(model)
    optimizer = torch.optim.SGD(twin.parameters(), lr=lr)
    values = []
    for data in batches:
        optimizer.zero_grad()
        guess = twin(data["x"]).reshape(-1)
        values.append((float(watched(guess, data["price"]).detach()), len(data["x"])))
        trained_with(guess, data["price"]).backward()
        optimizer.step()
    return sum(value * size for value, size in values) / sum(size for _, size in values)


def same_parameters(model, twin):
    return all(torch.allclose(a, b, atol=1e-6) for a, b in zip(model.parameters(), twin.parameters()))


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_turn_scope_is_the_alternating_turn():
    assert TURNS == ["/turn/kalfa/alternating"]


def test_alternating_declares_its_returns_mutations_bus_and_extras():
    facts = registry.facts("/turn/kalfa/alternating")
    assert facts.returns == ["models", "optimizers", "emas", "counters", "stream", "metrics"]
    assert facts.mutates == ("models", "optimizers", "emas", "counters")
    assert facts.bus == {"device": "device", "prep": "prep", "record": "record", "monitor": "monitor"}
    assert facts.get("extras") == ["amp", "grad_clip", "accumulate"]
    assert registry.aliases()["alternating"] == "/turn/kalfa/alternating"
    assert registry.aliases()["supervised"] == "/turn/kalfa/alternating"


def test_one_epoch_updates_once_per_batch_and_returns_the_running_mean():
    model = tiny_model()
    loader = loader_of()
    twin, mean, _, _ = replay(model, list(loader), build("/criterion/kalfa/mse"))
    optimizers = {"main": sgd("m", model, "mse")}
    out = turn({"m": model}, optimizers, loader, {"mse": criterion("/criterion/kalfa/mse")})
    assert set(out) == {"models", "optimizers", "emas", "counters", "stream", "metrics"}
    assert out["models"]["m"] is model and out["optimizers"] is optimizers
    assert out["counters"] == {"global_step": 4, "turn": 1}
    assert out["stream"] is None
    assert optimizers["main"].updates == 4
    assert same_parameters(model, twin)
    assert out["metrics"] == {"mse": pytest.approx(mean)}
    assert model.training is True


def test_accumulate_takes_several_batches_per_update():
    model = tiny_model()
    loader = loader_of()
    twin, mean, _, _ = replay(model, list(loader), build("/criterion/kalfa/mse"), accumulate=2)
    optimizers = {"main": sgd("m", model, "mse")}
    out = turn({"m": model}, optimizers, loader, {"mse": criterion("/criterion/kalfa/mse")}, extra={"accumulate": 2})
    assert out["counters"] == {"global_step": 2, "turn": 1}
    assert optimizers["main"].updates == 2
    assert same_parameters(model, twin)
    assert out["metrics"] == {"mse": pytest.approx(mean)}


def test_grad_clip_writes_the_gradient_norm_into_every_step_line(tmp_path):
    model = tiny_model()
    loader = loader_of()
    twin, _, norms, _ = replay(model, list(loader), build("/criterion/kalfa/mse"), grad_clip=0.5)
    monitor = Monitor()
    out = turn({"m": model}, {"main": sgd("m", model, "mse")}, loader, {"mse": criterion("/criterion/kalfa/mse")},
               extra={"grad_clip": 0.5}, record=str(tmp_path), monitor=monitor)
    lines = read_lines(tmp_path / "steps.jsonl")
    assert [set(line) for line in lines] == [{"step", "turn", "loss/main", "lr/main", "grad_norm/main"}] * 4
    assert [line["step"] for line in lines] == [1, 2, 3, 4]
    assert all(line["turn"] == 1 and line["lr/main"] == 0.1 for line in lines)
    assert [line["grad_norm/main"] for line in lines] == pytest.approx(norms)
    assert same_parameters(model, twin)
    assert monitor.begun == [4]
    assert monitor.lines == lines
    assert out["counters"]["global_step"] == 4


def test_step_lines_carry_the_loss_of_every_update(tmp_path):
    model = tiny_model()
    loader = loader_of(size=16)
    data = next(iter(loader))
    with torch.no_grad():
        before = float(build("/criterion/kalfa/mse")(model(data["x"]).reshape(-1), data["price"]))
    turn({"m": model}, {"main": sgd("m", model, "mse")}, loader, {"mse": criterion("/criterion/kalfa/mse")},
         record=str(tmp_path))
    lines = read_lines(tmp_path / "steps.jsonl")
    assert [set(line) for line in lines] == [{"step", "turn", "loss/main", "lr/main"}]
    assert lines[0]["loss/main"] == pytest.approx(before)


def test_order_and_steps_give_every_optimizer_its_place_and_count():
    first, second = tiny_model(index=0), tiny_model(index=1)
    started = copy.deepcopy(first), copy.deepcopy(second)
    loader = loader_of()
    trace = []
    optimizers = {"a": sgd("m1", first, "la"), "b": sgd("m2", second, "lb")}
    losses = {"la": objective("m1", trace), "lb": objective("m2", trace)}
    out = turn({"m1": first, "m2": second}, optimizers, loader, losses,
               params={"order": ["b", "a"], "steps": {"a": 1, "b": 2}}, predicts="m1")
    assert out["counters"] == {"global_step": 4, "turn": 1}
    assert optimizers["a"].updates == 4 and optimizers["b"].updates == 8
    assert [name for name, _ in trace] == ["m2", "m2", "m1"] * 4
    batches = list(loader)
    for position, (_, x) in enumerate(trace):
        assert torch.equal(x, batches[position // 3]["x"])
    doubled = [data for data in batches for _ in range(2)]
    twin, _, _, _ = replay(started[1], doubled, build("/criterion/kalfa/mse"))
    assert same_parameters(second, twin)
    lone, _, _, _ = replay(started[0], batches, build("/criterion/kalfa/mse"))
    assert same_parameters(first, lone)


def test_fresh_batch_takes_a_new_batch_for_every_update():
    first, second = tiny_model(index=0), tiny_model(index=1)
    loader = loader_of()
    trace = []
    optimizers = {"a": sgd("m1", first, "la"), "b": sgd("m2", second, "lb")}
    losses = {"la": objective("m1", trace), "lb": objective("m2", trace)}
    out = turn({"m1": first, "m2": second}, optimizers, loader, losses,
               params={"order": ["a", "b"], "steps": {"b": 2}, "fresh_batch": True}, predicts="m1")
    batches = list(loader)
    assert out["counters"] == {"global_step": 2, "turn": 1}
    assert optimizers["a"].updates == 2 and optimizers["b"].updates == 2
    assert [name for name, _ in trace] == ["m1", "m2", "m2", "m1"]
    for (_, x), expected in zip(trace, batches):
        assert torch.equal(x, expected["x"])


def test_order_must_name_defined_optimizers():
    model = tiny_model()
    with pytest.raises(KeyError, match=r"order names optimizers \['aux'\] that are not defined"):
        turn({"m": model}, {"main": sgd("m", model, "mse")}, loader_of(), {"mse": criterion("/criterion/kalfa/mse")},
             params={"order": ["main", "aux"]})


def test_an_optimizer_that_gets_no_batch_is_reported_idle():
    first, second = tiny_model(index=0), tiny_model(index=1)
    optimizers = {"a": sgd("m1", first, "la"), "b": sgd("m2", second, "lb")}
    losses = {"la": objective("m1"), "lb": objective("m2")}
    message = ("turn 1: optimizers ['b'] took no step; the train loader ran out of batches before their place in "
               "order ['a', 'b'] (steps {'a': 4}); more batches per turn or fewer steps for the optimizers before "
               "them")
    with pytest.warns(UserWarning, match=re.escape(message)):
        out = turn({"m1": first, "m2": second}, optimizers, loader_of(), losses,
                   params={"order": ["a", "b"], "steps": {"a": 4}, "fresh_batch": True}, predicts="m1")
    assert optimizers["a"].updates == 4 and optimizers["b"].updates == 0
    assert out["counters"]["global_step"] == 1


def test_a_loss_without_a_gradient_is_refused():
    model = tiny_model(trainable=False)
    with pytest.raises(ValueError, match=r"the loss of optimizer 'main' carries no gradient; its models are not "
                                          r"trainable or the loss does not depend on them"):
        turn({"m": model}, {"main": sgd("m", model, "mse")}, loader_of(), {"mse": criterion("/criterion/kalfa/mse")})


def test_an_optimizer_without_a_loss_is_refused():
    model = tiny_model()
    with pytest.raises(ValueError, match=r"optimizer 'main' names no loss"):
        turn({"m": model}, {"main": sgd("m", model, None)}, loader_of(), {"mse": criterion("/criterion/kalfa/mse")})


def test_a_loss_effect_swaps_what_the_optimizer_minimizes():
    model = tiny_model()
    loader = loader_of()
    twin, mae_mean, _, _ = replay(model, list(loader), build("/criterion/kalfa/mae"))
    losses = {"mse": criterion("/criterion/kalfa/mse"), "mae": criterion("/criterion/kalfa/mae")}
    out = turn({"m": model}, {"main": sgd("m", model, "mse")}, loader, losses, effects={"main.loss": "mae"})
    assert same_parameters(model, twin)
    assert set(out["metrics"]) == {"mse", "mae"}
    assert out["metrics"]["mae"] == pytest.approx(mae_mean)
    other = tiny_model()
    shadow, _, _, _ = replay(other, list(loader), build("/criterion/kalfa/mae"))
    turn({"m": other}, {"main": sgd("m", other, "mse")}, loader, losses, effects={"loss": "mae"})
    assert same_parameters(other, shadow)


def test_losses_not_minimized_and_metrics_are_observed_on_the_same_predictions():
    model = tiny_model()
    loader = loader_of()
    losses = {"mse": criterion("/criterion/kalfa/mse"), "mae": criterion("/criterion/kalfa/mae")}
    metrics = {"rmse": build("/adapter/kalfa/metric", metric=build("/metric/kalfa/rmse"))}
    mae_mean = observed_mean(model, list(loader), build("/criterion/kalfa/mse"), build("/criterion/kalfa/mae"))
    out = turn({"m": model}, {"main": sgd("m", model, "mse")}, loader, losses, metrics=metrics)
    assert set(out["metrics"]) == {"mse", "mae", "rmse"}
    assert out["metrics"]["rmse"] == pytest.approx(math.sqrt(out["metrics"]["mse"]))
    assert out["metrics"]["mae"] == pytest.approx(mae_mean)


def test_every_and_sets_keys_pace_the_observed_entries():
    model = tiny_model()
    losses = {"mse": criterion("/criterion/kalfa/mse"), "mae": criterion("/criterion/kalfa/mae")}
    metrics = {"rmse": build("/adapter/kalfa/metric", metric=build("/metric/kalfa/rmse"))}
    keys = {"mae": {"every": 2}}
    metric_keys = {"rmse": {"sets": ["valid"]}}
    first = turn({"m": model}, {"main": sgd("m", model, "mse")}, loader_of(), losses, metrics=metrics,
                 losses_keys=keys, metrics_keys=metric_keys)
    assert set(first["metrics"]) == {"mse"}
    second = turn({"m": model}, {"main": sgd("m", model, "mse")}, loader_of(), losses, metrics=metrics,
                  losses_keys=keys, metrics_keys=metric_keys, counters=first["counters"])
    assert set(second["metrics"]) == {"mse", "mae"}
    assert second["counters"] == {"global_step": 8, "turn": 2}


def test_steps_mode_keeps_a_stream_cursor_alive_across_turns():
    model = tiny_model()
    loader = loader_of()
    batches = list(loader)
    losses = {"mse": criterion("/criterion/kalfa/mse")}
    optimizers = {"main": sgd("m", model, "mse")}
    first = turn({"m": model}, optimizers, loader, losses, steps={"turn": 3, "total": 6})
    assert isinstance(first["stream"], Cursor) and first["stream"].endless
    assert first["counters"] == {"global_step": 3, "turn": 1}
    second = turn({"m": model}, optimizers, loader, losses, steps={"turn": 3, "total": 6}, stream=first["stream"],
                  counters=first["counters"])
    assert second["stream"] is first["stream"]
    assert second["counters"] == {"global_step": 6, "turn": 2}
    twin, _, _, _ = replay(tiny_model(), batches[:3] + [batches[3], batches[0], batches[1]],
                           build("/criterion/kalfa/mse"))
    assert same_parameters(model, twin)


def test_steps_mode_restarts_the_cursor_of_another_loader():
    model = tiny_model()
    losses = {"mse": criterion("/criterion/kalfa/mse")}
    first = turn({"m": model}, {"main": sgd("m", model, "mse")}, loader_of(), losses, steps={"turn": 2, "total": 4})
    other = loader_of(seed=1)
    second = turn({"m": model}, {"main": sgd("m", model, "mse")}, other, losses, steps={"turn": 2, "total": 4},
                  stream=first["stream"], counters=first["counters"])
    assert second["stream"] is not first["stream"]
    assert second["stream"].loader is other


def test_ema_copies_shift_after_every_update():
    model = tiny_model()
    ema = build("/lego/kalfa/clone", model=model, decay=0.5)
    loader = loader_of()
    twin, _, _, shadow = replay(model, list(loader), build("/criterion/kalfa/mse"), ema=ema.model)
    out = turn({"m": model}, {"main": sgd("m", model, "mse")}, loader, {"mse": criterion("/criterion/kalfa/mse")},
               emas={"m": ema})
    assert out["emas"]["m"] is ema
    assert same_parameters(model, twin)
    assert same_parameters(ema.model, shadow)


def test_the_turn_writes_no_steps_file_without_a_record(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = tiny_model()
    turn({"m": model}, {"main": sgd("m", model, "mse")}, loader_of(), {"mse": criterion("/criterion/kalfa/mse")})
    assert list(tmp_path.iterdir()) == []


def test_an_empty_train_loader_takes_no_step():
    model = tiny_model()
    dataset = build("/feed/kalfa/table", frame=frame(rows=3))
    loader = build("/loader/kalfa/torch", data=dataset, set="train", size=4, shuffle=False, drop_last=True)
    out = turn({"m": model}, {"main": sgd("m", model, "mse")}, loader, {"mse": criterion("/criterion/kalfa/mse")})
    assert out["counters"] == {"turn": 1}
    assert math.isnan(out["metrics"]["mse"])
