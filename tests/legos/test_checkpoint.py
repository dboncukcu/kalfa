import math

import pytest
import torch
from cirak.registry import registry

from helpers import build, tiny_model
from kalfa.std import STD_URIS
from kalfa.std.checkpoint.kalfa.policies import Best, Last, Snapshot
from kalfa.std.lego.kalfa.clone import Ema


POLICIES = sorted(uri for uri in STD_URIS if uri.startswith("/checkpoint/"))

PAYLOAD_KEYS = {"models", "optimizers", "emas", "counters", "rules", "checkpoint", "rng", "turn"}


def tags_of(policy, values, monitor="val/rmse"):
    return [policy.tags(None if value is None else {monitor: value}) for value in values]


def fill(model, value):
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(value)
    return model


def training_state(model, turn=2):
    optimizer = build("/optimizer/torch/sgd", models={"m": model}, params={"lr": 0.1})
    optimizer.torch()
    ema = build("/lego/kalfa/clone", model=model, decay=0.5)
    return {"models_next": {"m": model}, "optimizers_next": {"main": optimizer}, "emas_next": {"m": ema},
            "counters_next": {"turn": turn, "global_step": 4 * turn}, "rules_next": {"sticky": ["warm"]}}


def load(path):
    return torch.load(path, weights_only=False)


def checkpoint(state, policy, metrics, record):
    return build("/lego/kalfa/checkpoint", state=state, policy=policy, metrics=metrics, record=record)


def save_final(models, optimizers, emas, counters, rules, record, policy=None):
    return build("/lego/kalfa/save_final", models=models, optimizers=optimizers, emas=emas, counters=counters,
                 rules=rules, record=record, policy=policy)


def select(models, emas, which, record):
    return build("/lego/kalfa/select", models=models, emas=emas, which=which, record=record)


def init_state(state, epochs, steps, policy=None, resume=None):
    return build("/lego/kalfa/init_state", state=state, epochs=epochs, steps=steps, policy=policy, resume=resume)


def test_checkpoint_scope_is_the_three_kalfa_policies():
    assert POLICIES == ["/checkpoint/kalfa/best", "/checkpoint/kalfa/last", "/checkpoint/kalfa/snapshot"]


def test_every_policy_declares_the_files_it_writes():
    assert registry.facts("/checkpoint/kalfa/best").get("writes") == ["best", "last"]
    assert registry.facts("/checkpoint/kalfa/last").get("writes") == ["last"]
    assert registry.facts("/checkpoint/kalfa/snapshot").get("writes") == ["last", "snapshot"]
    for uri in POLICIES:
        assert registry.aliases()[uri.rsplit("/", 1)[1]] == uri


def test_best_tags_best_on_every_improvement_and_last_on_every_turn():
    policy = build("/checkpoint/kalfa/best", monitor="val/rmse", mode="min")
    assert isinstance(policy, Best)
    assert policy.monitor == "val/rmse"
    assert tags_of(policy, [1.0, 1.2, 0.8, 0.8, None, math.nan, 0.7]) == [
        ["best", "last"], ["last"], ["best", "last"], ["last"], ["last"], ["last"], ["best", "last"]]
    assert policy.state() == {"best": 0.7}


def test_best_mode_max_improves_upwards():
    policy = build("/checkpoint/kalfa/best", monitor="val/auroc", mode="max")
    assert tags_of(policy, [0.5, 0.4, 0.6], monitor="val/auroc") == [["best", "last"], ["last"], ["best", "last"]]


def test_best_without_last_writes_best_alone():
    policy = build("/checkpoint/kalfa/best", monitor="val/rmse", last=False)
    assert tags_of(policy, [1.0, 2.0, 0.5, None]) == [["best"], [], ["best"], []]


def test_best_refuses_an_unknown_mode():
    with pytest.raises(ValueError, match=r"mode must be min or max, got 'avg'"):
        build("/checkpoint/kalfa/best", monitor="val/rmse", mode="avg")


def test_best_restores_its_best_value_and_ignores_an_empty_state():
    policy = build("/checkpoint/kalfa/best", monitor="val/rmse")
    policy.restore({"best": 0.5})
    assert policy.best == 0.5
    assert tags_of(policy, [0.6, 0.4]) == [["last"], ["best", "last"]]
    policy.restore(None)
    policy.restore({})
    assert policy.best == 0.4


def test_last_tags_last_on_every_turn_without_state():
    policy = build("/checkpoint/kalfa/last")
    assert isinstance(policy, Last)
    assert policy.monitor is None
    assert tags_of(policy, [1.0, None, 3.0]) == [["last"], ["last"], ["last"]]
    assert policy.state() is None
    policy.restore({"anything": 1})


def test_snapshot_tags_a_numbered_snapshot_every_n_turns():
    policy = build("/checkpoint/kalfa/snapshot", every=2)
    assert isinstance(policy, Snapshot)
    assert tags_of(policy, [None] * 4) == [["last"], ["last", "snapshot_2"], ["last"], ["last", "snapshot_4"]]
    assert policy.state() == {"seen": 4}


def test_snapshot_resumes_its_count_from_the_state():
    policy = build("/checkpoint/kalfa/snapshot", every=2)
    policy.restore({"seen": 3})
    assert policy.tags({}) == ["last", "snapshot_4"]
    policy.restore(None)
    assert policy.tags({}) == ["last"]


def test_checkpoint_step_writes_the_policy_tags_with_the_full_payload(tmp_path):
    model = fill(tiny_model(), 1.0)
    state = training_state(model)
    policy = build("/checkpoint/kalfa/best", monitor="val/rmse")
    assert checkpoint(state, policy, {"val/rmse": 1.0}, str(tmp_path)) is None
    assert sorted(path.name for path in (tmp_path / "checkpoints").iterdir()) == ["best.pt", "last.pt"]
    payload = load(tmp_path / "checkpoints" / "best.pt")
    assert set(payload) == PAYLOAD_KEYS
    assert list(payload["models"]) == ["m"]
    assert all(torch.equal(payload["models"]["m"][key], value) for key, value in model.state_dict().items())
    assert set(payload["optimizers"]["main"]) == {"torch", "updates", "base"}
    assert list(payload["emas"]) == ["m"]
    assert payload["counters"] == {"turn": 2, "global_step": 8}
    assert payload["rules"] == {"sticky": ["warm"]}
    assert payload["checkpoint"] == {"best": 1.0}
    assert payload["turn"] == 2
    assert {"python", "numpy", "torch"} <= set(payload["rng"])
    checkpoint(state, policy, {"val/rmse": 2.0}, str(tmp_path))
    assert load(tmp_path / "checkpoints" / "best.pt")["checkpoint"] == {"best": 1.0}
    assert load(tmp_path / "checkpoints" / "last.pt")["checkpoint"] == {"best": 1.0}


def test_checkpoint_step_writes_nothing_without_a_policy_or_a_record(tmp_path):
    state = training_state(tiny_model())
    assert checkpoint(state, None, {"val/rmse": 1.0}, str(tmp_path)) is None
    assert checkpoint(state, build("/checkpoint/kalfa/last"), {"val/rmse": 1.0}, None) is None
    assert not (tmp_path / "checkpoints").exists()


def test_save_final_writes_the_final_state_with_the_policy_state(tmp_path):
    model = tiny_model()
    state = training_state(model, turn=3)
    policy = build("/checkpoint/kalfa/best", monitor="val/rmse")
    policy.tags({"val/rmse": 0.5})
    assert save_final(state["models_next"], state["optimizers_next"], state["emas_next"], state["counters_next"],
                      state["rules_next"], str(tmp_path), policy) is None
    payload = load(tmp_path / "final" / "state.pt")
    assert set(payload) == PAYLOAD_KEYS
    assert payload["checkpoint"] == {"best": 0.5}
    assert payload["turn"] == 3
    assert payload["rules"] == {"sticky": ["warm"]}
    assert save_final({}, {}, {}, {}, {}, None, policy) is None


def test_save_final_without_a_policy_writes_no_policy_state(tmp_path):
    state = training_state(tiny_model(), turn=3)
    save_final(state["models_next"], state["optimizers_next"], state["emas_next"], state["counters_next"],
               state["rules_next"], str(tmp_path))
    assert load(tmp_path / "final" / "state.pt")["checkpoint"] is None


def test_resuming_from_the_final_state_keeps_the_best_value(tmp_path):
    source = training_state(fill(tiny_model(), 1.0), turn=3)
    finished = build("/checkpoint/kalfa/best", monitor="val/rmse")
    finished.tags({"val/rmse": 0.5})
    save_final(source["models_next"], source["optimizers_next"], source["emas_next"], source["counters_next"],
               {"checkpoint": {"best": 0.9}}, str(tmp_path), finished)
    fresh = fill(tiny_model(), 0.0)
    state = {"models": {"m": fresh}, "emas": {}, "optimizers": {}, "counters": {}, "rules": {}}
    policy = build("/checkpoint/kalfa/best", monitor="val/rmse")
    assert init_state(state, 5, None, policy=policy, resume=str(tmp_path / "final" / "state.pt")) == 2
    assert policy.best == 0.5 and state["rules"]["checkpoint"] == {"best": 0.5}
    assert policy.tags({"val/rmse": 0.7}) == ["last"]


def test_select_best_loads_the_best_checkpoint_into_the_models(tmp_path):
    model = fill(tiny_model(), 1.0)
    state = training_state(model)
    checkpoint(state, build("/checkpoint/kalfa/best", monitor="val/rmse"), {"val/rmse": 1.0}, str(tmp_path))
    fill(model, 5.0)
    fill(state["emas_next"]["m"].model, 5.0)
    model.train()
    selected = select(state["models_next"], state["emas_next"], "best", str(tmp_path))
    assert list(selected) == ["m", "m.ema"]
    assert selected["m"] is model
    assert torch.equal(model.nodes["layer"].weight, torch.ones(1, 3))
    assert isinstance(selected["m.ema"], Ema)
    assert selected["m.ema"] is state["emas_next"]["m"]
    assert torch.equal(selected["m.ema"].model.nodes["layer"].weight, torch.ones(1, 3))
    assert all(not item.training for item in selected.values())


def test_select_last_reports_the_models_as_training_left_them_in_eval_mode(tmp_path):
    model = fill(tiny_model(), 2.0)
    model.train()
    selected = select({"m": model}, {}, "last", str(tmp_path))
    assert list(selected) == ["m"]
    assert selected["m"] is model
    assert torch.equal(model.nodes["layer"].weight, torch.full((1, 3), 2.0))
    assert model.training is False


def test_select_names_a_missing_best_checkpoint_and_an_unknown_report(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"report: best needs .*checkpoints/best.pt, but the best checkpoint "
                                                 r"was never written"):
        select({"m": tiny_model()}, {}, "best", str(tmp_path))
    with pytest.raises(ValueError, match=r"report must be best or last, got 'final'"):
        select({"m": tiny_model()}, {}, "final", str(tmp_path))


def test_init_state_counts_the_turns_left_from_epochs_or_steps():
    state = {"models": {"m": tiny_model()}, "emas": {}, "optimizers": {}, "counters": {}, "rules": {}}
    assert init_state(state, 3, None) == 3
    state["counters"]["turn"] = 1
    assert init_state(state, 3, None) == 2
    assert init_state(state, None, {"total": 10, "turn": 4}) == 2
    state["counters"]["turn"] = 9
    assert init_state(state, 3, None) == 0
    with pytest.raises(ValueError, match=r"training needs epochs or steps"):
        init_state(state, None, None)


def test_init_state_resumes_from_a_checkpoint_and_restores_the_policy(tmp_path):
    trained = fill(tiny_model(), 1.0)
    source = training_state(trained, turn=2)
    checkpoint(source, build("/checkpoint/kalfa/best", monitor="val/rmse"), {"val/rmse": 0.5}, str(tmp_path))
    fresh = fill(tiny_model(), 0.0)
    optimizer = build("/optimizer/torch/sgd", models={"m": fresh}, params={"lr": 0.1})
    state = {"models": {"m": fresh}, "emas": {"m": build("/lego/kalfa/clone", model=fresh, decay=0.5)},
             "optimizers": {"main": optimizer}, "counters": {}, "rules": {}}
    policy = build("/checkpoint/kalfa/best", monitor="val/rmse")
    left = init_state(state, 5, None, policy=policy, resume=str(tmp_path / "checkpoints" / "best.pt"))
    assert left == 3
    assert torch.equal(fresh.nodes["layer"].weight, torch.ones(1, 3))
    assert torch.equal(state["emas"]["m"].model.nodes["layer"].weight, torch.ones(1, 3))
    assert state["counters"] == {"turn": 2, "global_step": 8}
    assert state["rules"] == {"sticky": ["warm"], "checkpoint": {"best": 0.5}}
    assert policy.best == 0.5
    assert optimizer.pending is not None and optimizer.real is None
