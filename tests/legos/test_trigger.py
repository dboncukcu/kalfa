import functools
import math

from cirak.registry import registry

from helpers import build
from kalfa.std import STD_URIS


TRIGGERS = sorted(uri for uri in STD_URIS if uri.startswith("/trigger/"))


def run(trigger, values, monitor="val/loss", state=None):
    fired = []
    states = []
    for value in values:
        metrics = None if value is None else {monitor: value}
        hit, state = trigger(metrics, len(fired) + 1, state)
        fired.append(hit)
        states.append(dict(state))
    return fired, states


def test_trigger_scope_is_the_five_kalfa_triggers():
    assert TRIGGERS == ["/trigger/kalfa/after_turn", "/trigger/kalfa/metric_above", "/trigger/kalfa/metric_below",
                        "/trigger/kalfa/plateau", "/trigger/kalfa/time_budget"]


def test_every_trigger_is_a_partial_predicate_with_a_description_of_its_firing():
    for uri in TRIGGERS:
        facts = registry.facts(uri)
        assert facts.partial is True
        assert facts.kind == "predicate"
        assert registry.aliases()[uri.rsplit("/", 1)[1]] == uri
    assert registry.aliases()["after_epoch"] == "/trigger/kalfa/after_turn"
    assert registry.facts("/trigger/kalfa/after_turn").get("describe") == "turn ≥ {at}"
    assert registry.facts("/trigger/kalfa/time_budget").get("describe") == "after {minutes} minutes"
    assert registry.facts("/trigger/kalfa/metric_above").get("describe") == "{monitor} > {value}"
    assert registry.facts("/trigger/kalfa/metric_below").get("describe") == "{monitor} < {value}"
    assert registry.facts("/trigger/kalfa/plateau").get("describe") == "{monitor} plateau {patience}"
    assert isinstance(build("/trigger/kalfa/after_turn", at=2), functools.partial)


def test_after_turn_counts_the_turns_it_saw_and_stays_fired():
    trigger = build("/trigger/kalfa/after_turn", at=2)
    fired, states = run(trigger, [None, None, None])
    assert fired == [False, True, True]
    assert states == [{"seen": 1}, {"seen": 2}, {"seen": 3}]


def test_after_turn_resumes_from_the_seen_count_of_its_state():
    trigger = build("/trigger/kalfa/after_turn", at=3)
    assert trigger({}, 7, {"seen": 2}) == (True, {"seen": 3})
    assert trigger({}, 7, {"seen": 1}) == (False, {"seen": 2})
    assert trigger({}, 7, None) == (False, {"seen": 1})


def test_time_budget_fires_once_the_minutes_since_its_first_turn_have_passed(monkeypatch):
    clock = iter([100.0, 100.0 + 119.0, 100.0 + 120.0, 500.0])
    monkeypatch.setattr("kalfa.std.trigger.kalfa.clock.time.time", lambda: next(clock))
    trigger = build("/trigger/kalfa/time_budget", minutes=2)
    fired, states = run(trigger, [None, None, None, None])
    assert fired == [False, False, True, True]
    assert states == [{"started": 100.0}] * 4


def test_time_budget_of_zero_minutes_fires_at_its_first_turn():
    trigger = build("/trigger/kalfa/time_budget", minutes=0)
    hit, state = trigger({}, 1, {})
    assert hit is True
    assert set(state) == {"started"}


def test_metric_above_fires_while_the_value_is_above_the_bar():
    trigger = build("/trigger/kalfa/metric_above", monitor="val/auroc", value=0.8)
    fired, states = run(trigger, [0.5, 0.8, 0.9, 0.7, 0.85], monitor="val/auroc")
    assert fired == [False, False, True, False, True]
    assert states == [{}] * 5


def test_metric_below_fires_while_the_value_is_below_the_bar():
    trigger = build("/trigger/kalfa/metric_below", monitor="val/loss", value=1.0)
    fired, states = run(trigger, [1.5, 1.0, 0.9, 1.2, 0.1])
    assert fired == [False, False, True, False, True]


def test_metric_triggers_do_not_see_a_missing_or_nan_value():
    above = build("/trigger/kalfa/metric_above", monitor="val/auroc", value=0.5)
    below = build("/trigger/kalfa/metric_below", monitor="val/loss", value=0.5)
    for trigger in (above, below):
        assert trigger(None, 1, {"kept": 1}) == (False, {"kept": 1})
        assert trigger({"other": 0.1}, 1, None) == (False, {})
        assert trigger({trigger.keywords["monitor"]: math.nan}, 1, {}) == (False, {})


def test_metric_triggers_copy_their_state():
    trigger = build("/trigger/kalfa/metric_above", monitor="val/auroc", value=0.5)
    given = {"kept": 1}
    hit, state = trigger({"val/auroc": 0.9}, 1, given)
    assert hit is True and state == given and state is not given


def test_plateau_fires_after_patience_turns_without_improvement():
    trigger = build("/trigger/kalfa/plateau", monitor="val/loss", patience=2)
    fired, states = run(trigger, [1.0, 0.9, 0.95, 0.99, 0.8, 0.8])
    assert fired == [False, False, False, True, False, False]
    assert states == [{"best": 1.0, "wait": 0}, {"best": 0.9, "wait": 0}, {"best": 0.9, "wait": 1},
                      {"best": 0.9, "wait": 2}, {"best": 0.8, "wait": 0}, {"best": 0.8, "wait": 1}]


def test_plateau_with_no_patience_fires_at_every_turn_it_sees():
    trigger = build("/trigger/kalfa/plateau", monitor="val/loss", patience=0)
    fired, states = run(trigger, [1.0, 0.5, 0.7])
    assert fired == [True, True, True]
    assert states[-1] == {"best": 0.5, "wait": 1}


def test_plateau_mode_max_waits_for_a_rise():
    trigger = build("/trigger/kalfa/plateau", monitor="val/auroc", patience=1, mode="max")
    fired, states = run(trigger, [0.5, 0.7, 0.6, 0.71], monitor="val/auroc")
    assert fired == [False, False, True, False]
    assert states[-1] == {"best": 0.71, "wait": 0}


def test_plateau_min_delta_demands_a_large_enough_improvement():
    trigger = build("/trigger/kalfa/plateau", monitor="val/loss", patience=1, min_delta=0.05)
    fired, states = run(trigger, [1.0, 0.96, 0.9])
    assert fired == [False, True, False]
    assert states == [{"best": 1.0, "wait": 0}, {"best": 1.0, "wait": 1}, {"best": 0.9, "wait": 0}]


def test_plateau_does_not_count_turns_without_the_value():
    trigger = build("/trigger/kalfa/plateau", monitor="val/loss", patience=1)
    fired, states = run(trigger, [1.0, None, math.nan, 1.0])
    assert fired == [False, False, False, True]
    assert states == [{"best": 1.0, "wait": 0}, {"best": 1.0, "wait": 0}, {"best": 1.0, "wait": 0},
                      {"best": 1.0, "wait": 1}]
