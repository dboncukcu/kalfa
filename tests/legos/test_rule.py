import json
import logging

import pytest
from cirak.registry import registry

from helpers import build, tiny_model
from kalfa.std import STD_URIS


RULES = sorted(uri for uri in STD_URIS if uri.startswith("/rule/"))


def open_rules(rules):
    return build("/rule/kalfa/open", rules=rules)


def rule(rules, name, when, set, after=None, metrics=None, turn_index=None, sticky=True):
    return build("/rule/kalfa/rule", rules=rules, name=name, when=when, set=set, after=after, metrics=metrics,
                 turn_index=turn_index, sticky=sticky)


def effects(rules, models, optimizers, losses, loader=None):
    return build("/rule/kalfa/effects", rules=rules, models=models, optimizers=optimizers, losses=losses,
                 loader=loader)


def stop(rules, triggers, metrics=None, record=None):
    return build("/rule/kalfa/stop", rules=rules, triggers=triggers, metrics=metrics, record=record)


def above(value):
    return build("/trigger/kalfa/metric_above", monitor="v", value=value)


ALWAYS = above(0.0)
NEVER = above(10.0)
METRICS = {"v": 1.0}


def test_rule_scope_is_the_four_chain_steps():
    assert RULES == ["/rule/kalfa/effects", "/rule/kalfa/open", "/rule/kalfa/rule", "/rule/kalfa/stop"]


def test_chain_steps_declare_their_returns_and_bus():
    assert registry.facts("/rule/kalfa/rule").returns == "rules"
    assert registry.facts("/rule/kalfa/rule").bus == {"metrics": "metrics", "turn_index": "turn_index"}
    assert registry.facts("/rule/kalfa/effects").returns == ["effects", "losses", "models", "optimizers"]
    assert registry.facts("/rule/kalfa/effects").mutates == ("models", "optimizers")
    assert registry.facts("/rule/kalfa/stop").returns == ["rules", "stop"]
    assert registry.facts("/rule/kalfa/stop").bus == {"metrics": "metrics", "record": "record"}


def test_open_starts_the_turn_with_the_rules_that_fired_before_as_ready():
    assert open_rules({}) == {"fired": [], "pending": {}, "ready": []}
    assert open_rules(None) == {"fired": [], "pending": {}, "ready": []}
    previous = {"sticky": ["a"], "ever": ["a", "b"], "triggers": {"b": {"seen": 1}}, "fired": ["b"]}
    opened = open_rules(previous)
    assert opened == {"sticky": ["a"], "ever": ["a", "b"], "triggers": {"b": {"seen": 1}}, "fired": [],
                      "pending": {}, "ready": ["a", "b"]}
    assert previous["fired"] == ["b"]


def test_a_sticky_rule_fires_once_and_keeps_its_absolute_effects():
    turn_one = rule(open_rules({}), "warm", ALWAYS, {"main.lr": 0.5, "main.stem.lr": {"times": 0.5}},
                    metrics=METRICS, turn_index=1)
    assert turn_one["fired"] == ["warm"]
    assert turn_one["sticky"] == ["warm"]
    assert turn_one["ever"] == ["warm"]
    assert turn_one["triggers"] == {"warm": {}}
    assert turn_one["pending"] == {"main.lr": 0.5, "main.stem.lr": {"times": 0.5}}
    closed = stop(turn_one, [], metrics=METRICS)
    assert closed["stop"] is False
    assert closed["rules"]["effects"] == {"main.lr": 0.5, "main.stem.lr": {"times": 0.5}}
    assert "pending" not in closed["rules"] and "ready" not in closed["rules"]
    turn_two = rule(open_rules(closed["rules"]), "warm", NEVER, {"main.lr": 0.5, "main.stem.lr": {"times": 0.5}},
                    metrics=METRICS, turn_index=2)
    assert turn_two["fired"] == []
    assert turn_two["pending"] == {"main.lr": 0.5}
    assert stop(turn_two, [])["rules"]["effects"] == {"main.lr": 0.5}


def test_a_non_sticky_rule_is_asked_every_turn_and_its_trigger_starts_over():
    plateau = build("/trigger/kalfa/plateau", monitor="v", patience=1)
    turn_one = rule(open_rules({}), "cool", plateau, {"main.lr": {"times": 0.5}}, metrics=METRICS, sticky=False)
    assert turn_one["fired"] == [] and turn_one["triggers"] == {"cool": {"best": 1.0, "wait": 0}}
    assert turn_one["pending"] == {}
    turn_two = rule(open_rules(stop(turn_one, [])["rules"]), "cool", plateau, {"main.lr": {"times": 0.5}},
                    metrics=METRICS, sticky=False)
    assert turn_two["fired"] == ["cool"]
    assert turn_two["sticky"] == [] and turn_two["ever"] == ["cool"]
    assert turn_two["triggers"] == {"cool": {}}
    assert turn_two["pending"] == {"main.lr": {"times": 0.5}}
    turn_three = rule(open_rules(stop(turn_two, [])["rules"]), "cool", plateau, {"main.lr": {"times": 0.5}},
                      metrics=METRICS, sticky=False)
    assert turn_three["fired"] == []
    assert turn_three["triggers"] == {"cool": {"best": 1.0, "wait": 0}}
    assert turn_three["pending"] == {}


def test_a_rule_waits_for_its_after_rule_to_have_fired_in_an_earlier_turn():
    turn_one = rule(open_rules({}), "first", ALWAYS, {"main.lr": 0.1}, metrics=METRICS)
    turn_one = rule(turn_one, "second", ALWAYS, {"main.lr": 0.2}, after="first", metrics=METRICS)
    assert turn_one["fired"] == ["first"]
    assert "second" not in turn_one["triggers"]
    assert turn_one["pending"] == {"main.lr": 0.1}
    turn_two = open_rules(stop(turn_one, [])["rules"])
    assert turn_two["ready"] == ["first"]
    turn_two = rule(turn_two, "first", ALWAYS, {"main.lr": 0.1}, metrics=METRICS)
    turn_two = rule(turn_two, "second", ALWAYS, {"main.lr": 0.2}, after="first", metrics=METRICS)
    assert turn_two["fired"] == ["second"]
    assert turn_two["sticky"] == ["first", "second"]
    assert turn_two["pending"] == {"main.lr": 0.2}


def test_a_non_sticky_after_rule_readies_the_ones_behind_it():
    turn_one = rule(open_rules({}), "pulse", ALWAYS, {"a.lr": 1.0}, metrics=METRICS, sticky=False)
    turn_two = open_rules(stop(turn_one, [])["rules"])
    assert turn_two["ready"] == ["pulse"]
    turn_two = rule(turn_two, "pulse", NEVER, {"a.lr": 1.0}, metrics=METRICS, sticky=False)
    turn_two = rule(turn_two, "next", ALWAYS, {"b.lr": 2.0}, after="pulse", metrics=METRICS)
    assert turn_two["fired"] == ["next"]


def test_later_rules_win_the_same_key():
    turn_one = rule(open_rules({}), "first", ALWAYS, {"main.lr": 0.1, "m.trainable": False}, metrics=METRICS)
    turn_one = rule(turn_one, "second", ALWAYS, {"main.lr": 0.2}, metrics=METRICS)
    assert turn_one["fired"] == ["first", "second"]
    assert turn_one["pending"] == {"main.lr": 0.2, "m.trainable": False}


def test_a_rule_logs_its_firing_with_the_effects(caplog):
    caplog.set_level(logging.INFO, logger="kalfa")
    rule(open_rules({}), "warm", ALWAYS, {"main.lr": 0.5, "ws.terms": {"a": 2.0}}, metrics=METRICS)
    assert "warm fired: main.lr=0.5, ws.terms={'a': 2.0}" in caplog.text


def test_stop_ors_the_triggers_and_keeps_their_states():
    triggers = [build("/trigger/kalfa/after_turn", at=2), build("/trigger/kalfa/metric_below", monitor="v", value=0.0)]
    first = stop(open_rules({}), triggers, metrics=METRICS)
    assert first["stop"] is False
    assert first["rules"]["stop"] == [{"seen": 1}, {}]
    assert first["rules"]["stop_fired"] == []
    second = stop(open_rules(first["rules"]), triggers, metrics=METRICS)
    assert second["stop"] is True
    assert second["rules"]["stop"] == [{"seen": 2}, {}]
    assert second["rules"]["stop_fired"] == [0]
    assert second["rules"]["effects"] == {}


def test_stop_json_in_the_record_ends_the_loop(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="kalfa")
    assert stop(open_rules({}), [], record=str(tmp_path))["stop"] is False
    (tmp_path / "stop.json").write_text(json.dumps({"by": "kalfa stop", "at": "noon"}))
    assert stop(open_rules({}), [], record=str(tmp_path))["stop"] is True
    assert "stopping after this turn: stop requested by kalfa stop at noon" in caplog.text
    (tmp_path / "stop.json").write_text("")
    assert stop(open_rules({}), [], record=str(tmp_path))["stop"] is True
    assert stop(open_rules({}), [], record=None)["stop"] is False


def test_effects_reach_models_optimizers_and_losses_before_the_turn():
    model = tiny_model()
    stem = tiny_model(index=1)
    optimizer = build("/optimizer/torch/sgd", models={"m": model, "stem": stem},
                      params={"lr": 0.1, "groups": [{"name": "stem", "match": "stem.*", "lr": 0.2}]}, loss="mse")
    losses = {"huber": build("/adapter/kalfa/criterion", criterion=build("/criterion/kalfa/huber", delta=1.0)),
              "ws": build("/adapter/kalfa/objective",
                          objective=build("/objective/kalfa/weighted_sum", terms={"a": 1.0, "b": 0.5}))}
    found = {"m.trainable": False, "main.lr": 0.05, "main.stem.lr": {"times": 0.5}, "huber.delta": 3.0,
             "ws.terms.a": 2.0, "main.loss": "huber", "loss": "ws"}
    rules = {"effects": dict(found), "sticky": ["warm"]}
    out = effects(rules, {"m": model, "stem": stem}, {"main": optimizer}, losses)
    assert set(out) == {"effects", "losses", "models", "optimizers"}
    assert out["effects"] == found
    assert out["models"] == {"m": model, "stem": stem} and out["optimizers"] == {"main": optimizer}
    assert model.trainable is False
    assert all(parameter.requires_grad is False for parameter in model.parameters())
    assert stem.trainable is True
    assert optimizer.params["lr"] == 0.05
    assert optimizer.rates() == {"stem": 0.1}
    assert out["losses"] is not losses
    assert out["losses"]["huber"] is not losses["huber"]
    assert out["losses"]["huber"].criterion.keywords == {"delta": 3.0}
    assert losses["huber"].criterion.keywords == {"delta": 1.0}
    assert out["losses"]["ws"].objective.keywords == {"terms": {"a": 2.0, "b": 0.5}}
    assert losses["ws"].objective.keywords == {"terms": {"a": 1.0, "b": 0.5}}


def test_effects_without_any_effect_return_a_copy_of_the_losses():
    losses = {"mse": build("/adapter/kalfa/criterion", criterion=build("/criterion/kalfa/mse"))}
    out = effects({}, {}, {}, losses)
    assert out["effects"] == {}
    assert out["losses"] == losses and out["losses"] is not losses


def test_an_effect_must_name_a_model_an_optimizer_or_a_loss():
    with pytest.raises(KeyError, match=r"rule effect 'zz.lr' names neither a model, an optimizer nor a loss"):
        effects({"effects": {"zz.lr": 0.1}}, {"m": tiny_model()}, {}, {})


def test_a_trainable_effect_restores_the_gradients_of_a_frozen_model():
    model = tiny_model(trainable=False)
    assert all(parameter.requires_grad is False for parameter in model.parameters())
    effects({"effects": {"m.trainable": True}}, {"m": model}, {}, {})
    assert model.trainable is True
    assert all(parameter.requires_grad is True for parameter in model.parameters())
