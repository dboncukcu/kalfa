"""Checkpoint policies, files, resume through init_state, final/ and the report selection."""

import pytest
import torch

import kalfa  # noqa: F401
from helpers import tiny_model
from kalfa.std.checkpoint import best, checkpoint, init_state, last, load, save_final, select, snapshot
from kalfa.std.optimizer import sgd


def state_of(seed=1):
    model = tiny_model(seed=seed)
    optimizer = sgd({"model": model}, {"lr": 0.1}, None, "l")
    return {"models": {"model": model}, "optimizers": {"model": optimizer}, "emas": {},
            "counters": {"global_step": 0, "turn": 0}, "rules": {}}


def suffixed(state):
    return {f"{key}_next": value for key, value in state.items()}


def test_policies():
    policy = best("val/rmse")
    assert policy.tags({"val/rmse": 1.0}) == ["best", "last"]
    assert policy.tags({"val/rmse": 2.0}) == ["last"]
    assert policy.tags({}) == ["last"] and policy.tags({"val/rmse": float("nan")}) == ["last"]
    assert policy.tags({"val/rmse": 0.5}) == ["best", "last"] and policy.state() == {"best": 0.5}
    up = best("val/acc", mode="max")
    up.tags({"val/acc": 0.5})
    assert up.tags({"val/acc": 0.6}) == ["best", "last"]
    with pytest.raises(ValueError):
        best("x", mode="sideways")
    assert last().tags({}) == ["last"]
    snap = snapshot(2)
    assert snap.tags({}) == ["last"] and snap.tags({}) == ["last", "snapshot_2"]


def test_checkpoint_writes_files_and_restores_the_policy(tmp_path):
    state = state_of()
    state["models"]["model"](torch.ones(1, 3)).sum().backward()
    state["optimizers"]["model"].step()
    policy = best("val/rmse")
    checkpoint(suffixed(state), policy, {"val/rmse": 1.0}, str(tmp_path))
    assert (tmp_path / "checkpoints" / "best.pt").exists() and (tmp_path / "checkpoints" / "last.pt").exists()
    checkpoint(suffixed(state), policy, {"val/rmse": 3.0}, str(tmp_path))
    data = load(tmp_path / "checkpoints" / "last.pt")
    assert set(data) >= {"models", "optimizers", "counters", "rules", "rng", "checkpoint", "turn"}
    assert data["checkpoint"] == {"best": 1.0} and "model" in data["models"]
    assert checkpoint(suffixed(state), None, {}, str(tmp_path)) is None
    fresh = best("val/rmse")
    state["rules"] = {"checkpoint": {"best": 0.2}}
    checkpoint(suffixed(state), fresh, {"val/rmse": 0.5}, str(tmp_path))
    assert fresh.best == 0.2


def test_init_state_moves_resumes_and_counts_the_turns_left(tmp_path):
    state = state_of()
    assert init_state(state, epochs=10, steps=None, device="cpu") == 10
    state["models"]["model"](torch.ones(1, 3)).sum().backward()
    state["optimizers"]["model"].step()
    state["counters"] = {"global_step": 7, "turn": 4}
    state["rules"] = {"sticky": ["a"]}
    torch.manual_seed(11)
    checkpoint(suffixed(state), last(), {}, str(tmp_path))
    marker = torch.rand(1)
    fresh = state_of(seed=5)
    left = init_state(fresh, epochs=10, steps=None, resume=str(tmp_path / "checkpoints" / "last.pt"), device="cpu")
    assert left == 6
    assert fresh["counters"] == {"global_step": 7, "turn": 4} and fresh["rules"]["sticky"] == ["a"]
    assert torch.equal(fresh["models"]["model"].nodes["layer"].weight, state["models"]["model"].nodes["layer"].weight)
    assert torch.equal(torch.rand(1), marker)
    assert init_state(state_of(), epochs=None, steps={"total": 25, "turn": 10}, device="cpu") == 3
    with pytest.raises(ValueError):
        init_state(state_of(), epochs=None, steps=None)


def test_save_final_and_select(tmp_path):
    state = state_of()
    save_final(state["models"], state["optimizers"], state["emas"], state["counters"], state["rules"], str(tmp_path))
    assert (tmp_path / "final" / "state.pt").exists()
    chosen = select(state["models"], {}, "last", str(tmp_path))
    assert chosen["model"] is not state["models"]["model"] and not chosen["model"].training
    with pytest.raises(FileNotFoundError):
        select(state["models"], {}, "best", str(tmp_path))
    checkpoint(suffixed(state), best("v"), {"v": 1.0}, str(tmp_path))
    with torch.no_grad():
        state["models"]["model"].nodes["layer"].weight.fill_(9.0)
    chosen = select(state["models"], {}, "best", str(tmp_path))
    assert float(chosen["model"].nodes["layer"].weight.max().detach()) < 9.0
    with pytest.raises(ValueError):
        select(state["models"], {}, "middle", str(tmp_path))
