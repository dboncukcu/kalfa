"""Triggers, the rule chain (after, sticky, last wins) and the stop decision."""

import functools

import kalfa  # noqa: F401
from kalfa.std.rule import effects, open, rule, stop
from kalfa.std.trigger import after_turn, metric_above, metric_below, plateau, time_budget


def test_after_turn_counts_updates_not_the_index():
    state = {}
    fired = []
    for _ in range(3):
        hit, state = after_turn({}, None, state, at=2)
        fired.append(hit)
    assert fired == [False, True, True] and state == {"seen": 3}


def test_metric_triggers_ignore_missing_values():
    assert metric_below({"a": 0.5}, 0, {}, monitor="a", value=1.0) == (True, {})
    assert metric_below({"a": 2.0}, 0, {}, monitor="a", value=1.0) == (False, {})
    assert metric_below({}, 0, {}, monitor="a", value=1.0) == (False, {})
    assert metric_below({"a": float("nan")}, 0, {}, monitor="a", value=1.0) == (False, {})
    assert metric_above({"a": 2.0}, 0, {}, monitor="a", value=1.0) == (True, {})


def test_plateau_counts_only_turns_with_the_value():
    state = {}
    outcomes = []
    for value in (1.0, 0.9, None, 0.95, 0.96, 0.97):
        metrics = {} if value is None else {"v": value}
        hit, state = plateau(metrics, 0, state, monitor="v", patience=3)
        outcomes.append(hit)
    assert outcomes == [False, False, False, False, False, True]
    assert state == {"best": 0.9, "wait": 3}
    hit, state = plateau({"v": 5.0}, 0, {}, monitor="v", patience=1, mode="max")
    hit, state = plateau({"v": 4.0}, 0, state, monitor="v", patience=1, mode="max")
    assert hit and state["best"] == 5.0
    hit, state = plateau({"v": 0.89}, 0, {"best": 0.9, "wait": 0}, monitor="v", patience=2, min_delta=0.05)
    assert state["wait"] == 1


def test_time_budget():
    hit, state = time_budget({}, 0, {}, minutes=1000)
    assert not hit and "started" in state
    hit, _ = time_budget({}, 0, {"started": 0}, minutes=0)
    assert hit


def trig(hit):
    return lambda metrics, turn_index, state: (hit, {**state, "seen": state.get("seen", 0) + 1})


def chain(rules, specs, metrics=None, turn_index=0):
    current = open(rules)
    for spec in specs:
        current = rule(current, spec["name"], spec["when"], spec["set"], spec.get("after"), metrics, turn_index)
    return stop(current, [], metrics)


def test_after_sticky_and_last_wins():
    specs = [{"name": "a", "when": trig(True), "set": {"loss": "x"}},
             {"name": "b", "when": trig(True), "set": {"loss": "y"}, "after": "a"},
             {"name": "c", "when": trig(False), "set": {"loss": "z"}, "after": "b"}]
    rules = {}
    out = chain(rules, specs)
    rules = out["rules"]
    assert rules["fired"] == ["a"] and rules["sticky"] == ["a"] and rules["effects"] == {"loss": "x"}
    assert "b" not in rules["triggers"]
    out = chain(rules, specs)
    rules = out["rules"]
    assert rules["fired"] == ["b"] and rules["sticky"] == ["a", "b"] and rules["effects"] == {"loss": "y"}
    assert rules["triggers"]["a"]["seen"] == 1
    specs[2]["when"] = trig(True)
    rules = chain(rules, specs)["rules"]
    assert rules["fired"] == ["c"] and rules["effects"] == {"loss": "z"}
    rules = chain(rules, specs)["rules"]
    assert rules["fired"] == [] and rules["effects"] == {"loss": "z"}
    assert effects(rules) == {"loss": "z"}


def test_list_order_wins_over_firing_order():
    specs = [{"name": "first", "when": trig(False), "set": {"loss": "x"}},
             {"name": "second", "when": trig(True), "set": {"loss": "y"}}]
    rules = chain({}, specs)["rules"]
    assert rules["effects"] == {"loss": "y"}
    specs[0]["when"] = trig(True)
    rules = chain(rules, specs)["rules"]
    assert rules["effects"] == {"loss": "y"} and rules["sticky"] == ["second", "first"]


def test_stop_is_an_or_and_keeps_trigger_states():
    rules = open({})
    out = stop(rules, [functools.partial(metric_below, monitor="v", value=1.0), trig(False)], {"v": 0.5})
    assert out["stop"] is True and out["rules"]["stop_fired"] == [0] and len(out["rules"]["stop"]) == 2
    out = stop(out["rules"], [functools.partial(metric_below, monitor="v", value=1.0), trig(False)], {"v": 5.0})
    assert out["stop"] is False and out["rules"]["stop"][1] == {"seen": 2}
    assert stop(open({}), [], {})["stop"] is False
    assert effects({}) == {}
