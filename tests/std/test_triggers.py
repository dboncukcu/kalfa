import math
from types import SimpleNamespace

import pytest

from kalfa.core import aliases
from kalfa.std import triggers
from kalfa.std.triggers import Above, AfterEpoch, Below, Every, Plateau, TimeBudget


def ctx_at(turn, **latest):
    return SimpleNamespace(turn=turn, latest=latest)


def seen(trigger, turn, value):
    return trigger.fires(ctx_at(turn, **{trigger.watch: value}))


def test_plateau_fires_after_patience_turns_without_improvement():
    trigger = Plateau("val/rmse", patience=2)
    values = [1.0, 0.9, 0.95, 0.91, 0.5, 0.6, 0.7]
    fired = [seen(trigger, turn, value) for turn, value in enumerate(values, 1)]
    assert fired == [False, False, False, True, False, False, True]
    assert trigger.best == 0.5
    assert trigger.describe() == "val/rmse plateau 2"


def test_plateau_skips_missing_and_nan_values():
    trigger = Plateau("val/rmse", patience=1)
    assert seen(trigger, 1, 1.0) is False
    assert trigger.fires(ctx_at(2)) is False
    assert seen(trigger, 3, math.nan) is False
    assert trigger.wait == 0
    assert seen(trigger, 4, 1.0) is True


def test_plateau_max_mode_and_min_delta():
    trigger = Plateau("val/f1", patience=1, mode="max", min_delta=0.05)
    assert not seen(trigger, 1, 0.5)
    assert seen(trigger, 2, 0.52)
    assert not seen(trigger, 3, 0.6)
    with pytest.raises(ValueError):
        Plateau("val/f1", patience=1, mode="up")


def test_plateau_state_survives_a_checkpoint():
    trigger = Plateau("val/rmse", patience=3)
    seen(trigger, 1, 1.0)
    seen(trigger, 2, 1.2)
    state = trigger.state_dict()
    assert state == {"best": 1.0, "wait": 1}
    fresh = Plateau("val/rmse", patience=3)
    fresh.load_state_dict(state)
    assert not seen(fresh, 3, 1.1)
    assert seen(fresh, 4, 1.1)
    fresh.reset()
    assert fresh.best is None and fresh.wait == 0


def test_above_and_below_read_latest():
    assert seen(Above("train/huber", 0.5), 1, 0.7)
    assert not seen(Above("train/huber", 0.5), 1, 0.5)
    assert not Above("train/huber", 0.5).fires(ctx_at(1))
    assert seen(Below("train/huber", 0.01), 1, 0.001)
    assert not seen(Below("train/huber", 0.01), 1, math.nan)
    assert Below("train/huber", 0.01).describe() == "train/huber < 0.01"
    assert Above("train/huber", 0.5).describe() == "train/huber > 0.5"


def test_time_budget_counts_from_the_first_call(monkeypatch):
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(triggers, "time", SimpleNamespace(time=lambda: clock.now))
    trigger = TimeBudget(minutes=2)
    assert not trigger.fires(ctx_at(1))
    assert trigger.started == 1000.0
    clock.now = 1119.0
    assert not trigger.fires(ctx_at(2))
    clock.now = 1120.0
    assert trigger.fires(ctx_at(3))
    assert trigger.state_dict() == {"started": 1000.0}
    assert trigger.describe() == "after 2.0 minutes"
    trigger.reset()
    assert trigger.started is None


def test_after_epoch_and_every_count_turns():
    assert [AfterEpoch(3).fires(ctx_at(turn)) for turn in (1, 2, 3, 4)] == [False, False, True, True]
    assert [Every(2).fires(ctx_at(turn)) for turn in (1, 2, 3, 4)] == [False, True, False, True]
    assert not Every(0).fires(ctx_at(4))


def test_short_names_are_registered():
    table = aliases.ALIASES
    for name in ("after_epoch", "every", "plateau", "above", "below", "time_budget"):
        assert name in table or any(getattr(entry, "name", None) == name for entry in table.values()), name
