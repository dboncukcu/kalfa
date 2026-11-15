import json
import logging
from types import SimpleNamespace

import pytest
from torch import nn

from kalfa.core.context import Context, TrainSettings
from kalfa.core.errors import RunError
from kalfa.core.record import RecordWriter
from kalfa.core.rules import Patch, Rule, RuleSpec, apply_patch, parse_rules, replay, resolve_value, tokenize


RULES_01 = {
    "to_huber": {"when": {"use": "after_epoch", "at": 50}, "set": {"optimizers.main.minimizes": "@losses.huber"}},
    "to_mae": {"after": "to_huber", "when": {"use": "below", "watch": "train/huber", "value": 0.01},
               "set": {"optimizers.main.minimizes": "@losses.mae"}},
    "stop": {"when": {"use": "plateau", "watch": "val/rmse", "patience": 6}, "stop": True},
}


class Holder:
    def __init__(self, **attributes):
        self.__dict__.update(attributes)


class FakeOptimizer:
    def __init__(self, lr, minimizes=None):
        self.params = {"lr": lr}
        self.minimizes = minimizes
        self.calls = []

    def set_param(self, name, value):
        self.params[name] = value
        self.calls.append((name, value))


class FakeModel:
    def __init__(self, module):
        self.module = module
        self.ema = Holder(decay=0.999)
        self.freeze = False
        self.frozen = []

    def set_freeze(self, value):
        self.freeze = value
        self.frozen.append(value)


class FakeTrigger:
    def __init__(self, answer=True):
        self.answer = answer
        self.resets = 0

    def fires(self, ctx):
        return self.answer

    def reset(self):
        self.resets += 1

    def describe(self):
        return "fake trigger"


class FakeAction:
    def __init__(self):
        self.calls = 0

    def __call__(self, ctx):
        self.calls += 1


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Linear(2, 2)


def make_plan():
    return {
        "nodes": {
            "optimizers.main": {"slot": "optimizer", "params": {"lr": 1e-3},
                                "keys": {"minimizes": {"ref": "losses.mse"}}},
            "losses.total": {"slot": "loss", "keys": {},
                             "params": {"constraints": [{"of": {"ref": "losses.mae"}, "epsilon": 0.05}]}},
            "losses.huber": {"slot": "loss", "params": {"delta": 1.0}, "keys": {}},
            "models.net": {"slot": "model", "params": {"hidden": [8]}, "keys": {}},
            "train.step": {"slot": "step", "params": {}, "keys": {}},
        },
        "train": {"epochs": 10, "updates": None},
    }


def make_ctx(record=None, plan=None):
    tree = SimpleNamespace(config=None, settings={}, nodes={})
    ctx = Context(tree, plan)
    ctx.models = {"net": FakeModel(Net())}
    ctx.losses = {"mse": Holder(kind="mse"), "huber": Holder(delta=1.0), "wrapped": Holder(function=Holder(delta=1.0)),
                  "total": Holder(constraints=[Holder(epsilon=0.05)])}
    ctx.metrics = {"rmse": Holder(metric=Holder(squared=False), every=1)}
    ctx.optimizers = {"main": FakeOptimizer(1e-3, ctx.losses["mse"])}
    ctx.data = Holder(n_classes=3)
    ctx.train = TrainSettings(epochs=10)
    ctx.step = Holder(accumulate=1)
    ctx.record = record
    ctx.turn = 3
    return ctx


def events_in(directory):
    path = directory / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def codes(problems):
    return [problem.code for problem in problems]


def test_parse_rules_keeps_config_order_and_fields():
    problems = []
    specs = parse_rules(RULES_01, problems)
    assert problems == []
    assert [spec.name for spec in specs] == ["to_huber", "to_mae", "stop"]
    assert specs[0].when == {"use": "after_epoch", "at": 50}
    assert specs[0].set == {"optimizers.main.minimizes": "@losses.huber"}
    assert specs[0].after is None and specs[0].once is True and specs[0].stop is False and specs[0].run is None
    assert specs[1].after == "to_huber"
    assert specs[2].stop is True and specs[2].set == {}
    assert parse_rules(None, problems) == []
    assert problems == []


def test_parse_rules_accepts_once_run_and_patch_values():
    raw = {"samples": {"when": {"use": "every", "epochs": 5}, "once": False,
                       "run": {"use": "write_samples", "sampler": "@generate", "n": 16}},
           "warm": {"set": {"losses.elbo.w_rec": {"times": 0.5}, "losses.elbo.recon": "@losses.huber",
                            "train.epochs": {"plus": 10}, "models.net.freeze": ["backbone.*"],
                            "losses.total.constraints[0].epsilon": 0.02}}}
    problems = []
    specs = parse_rules(raw, problems)
    assert problems == []
    assert specs[0].once is False and specs[0].run == raw["samples"]["run"]
    assert list(specs[1].set) == list(raw["warm"]["set"])


def test_parse_rules_reports_problems():
    raw = {
        "typo": {"when": {"use": "every", "epochs": 1}, "sets": {"a.b": 1}},
        "bad_set": {"set": ["a.b"]},
        "nothing": {"when": {"use": "every", "epochs": 1}},
        "orphan": {"after": "ghost", "stop": True},
        "selfish": {"after": "selfish", "stop": True},
        "bad_path": {"set": {"losses.total.constraints[x].epsilon": 1}},
        "bad_value": {"set": {"losses.a.w": {"use": "foo"}, "losses.a.v": {"times": "two"}, "losses.a.u": {"a": 1}}},
        "bad_once": {"once": "yes", "stop": True},
        "bad_stop": {"stop": "yes"},
        "bad_when": {"when": [1, 2], "stop": True},
        "bad_body": 5,
    }
    problems = []
    specs = parse_rules(raw, problems)
    assert [spec.name for spec in specs] == [name for name in raw if name != "bad_body"]
    by_path = {(problem.code, problem.path) for problem in problems}
    assert ("rule_key", "rules.typo.sets") in by_path
    assert ("rule_empty", "rules.typo") in by_path
    assert ("rule_set", "rules.bad_set.set") in by_path
    assert ("rule_empty", "rules.bad_set") not in by_path
    assert ("rule_empty", "rules.nothing") in by_path
    assert ("rule_after", "rules.orphan.after") in by_path
    assert ("rule_after", "rules.selfish.after") in by_path
    assert ("rule_path", "rules.bad_path.set") in by_path
    assert codes(problems).count("rule_value") == 6
    assert ("rule_value", "rules.bad_once.once") in by_path
    assert ("rule_value", "rules.bad_stop.stop") in by_path
    assert ("rule_value", "rules.bad_when.when") in by_path
    assert ("rule_type", "rules.bad_body") in by_path
    assert all(problem.severity == "error" for problem in problems)
    other = []
    assert parse_rules(["stop"], other) == [] and codes(other) == ["rules_type"]


def test_tokenize():
    assert tokenize("losses.total.constraints[0].epsilon") == ["losses", "total", "constraints", 0, "epsilon"]
    assert tokenize("optimizers.main.lr") == ["optimizers", "main", "lr"]
    assert tokenize("a[0][12].b") == ["a", 0, 12, "b"]
    assert tokenize(" train.epochs ") == ["train", "epochs"]
    for bad in ("", "   ", "a..b", "a[", "a[x]", "a[0]b", ".a", "a.", "a[-1]", 7, None):
        with pytest.raises(RunError):
            tokenize(bad)


def test_resolve_value():
    ctx = make_ctx()
    assert resolve_value(ctx, 2.0, {"times": 0.5}) == 1.0
    assert resolve_value(ctx, 3, {"plus": 4}) == 7
    assert resolve_value(ctx, 3, 9) == 9
    assert resolve_value(ctx, None, "plain") == "plain"
    assert resolve_value(ctx, None, [1, 2]) == [1, 2]
    assert resolve_value(ctx, None, "@losses.huber") is ctx.losses["huber"]
    assert resolve_value(ctx, None, "@models.net") is ctx.models["net"].module
    assert resolve_value(ctx, None, "@models.net.ema") is ctx.models["net"].ema
    assert resolve_value(ctx, None, "@models.net.decoder") is ctx.models["net"].module.decoder
    assert resolve_value(ctx, None, "@optimizers.main") is ctx.optimizers["main"]
    assert resolve_value(ctx, None, "@data.n_classes") == 3
    assert resolve_value(ctx, None, {"ref": "losses.huber"}) is ctx.losses["huber"]
    assert resolve_value(ctx, None, {"ref": "models.net", "attr": "ema"}) is ctx.models["net"].ema
    for bad in ("@losses.nope", "@nowhere.x", "@losses", "@models.net.nope", "@generate", "@data.missing"):
        with pytest.raises(RunError):
            resolve_value(ctx, None, bad)
    with pytest.raises(RunError):
        resolve_value(ctx, None, {"times": 2})


def test_apply_patch_on_attributes_dicts_and_lists():
    ctx = make_ctx()
    patch = apply_patch(ctx, "losses.huber.delta", 0.5, by="r", turn=7)
    assert patch == Patch("losses.huber.delta", 0.5, 0.5, "r", 7)
    assert ctx.losses["huber"].delta == 0.5
    assert apply_patch(ctx, "losses.wrapped.delta", {"times": 2}).value == 2.0
    assert ctx.losses["wrapped"].function.delta == 2.0
    assert apply_patch(ctx, "metrics.rmse.squared", True).turn == 3
    assert ctx.metrics["rmse"].metric.squared is True
    apply_patch(ctx, "metrics.rmse.every", 2)
    assert ctx.metrics["rmse"].every == 2
    apply_patch(ctx, "losses.total.constraints[0].epsilon", 0.02)
    assert ctx.losses["total"].constraints[0].epsilon == 0.02
    apply_patch(ctx, "train.step.accumulate", 4)
    assert ctx.step.accumulate == 4
    ctx.train = TrainSettings(updates={"total": 100, "turn": 10})
    apply_patch(ctx, "train.updates.total", {"plus": 50})
    assert ctx.train.updates == {"total": 150, "turn": 10}
    ctx.train = TrainSettings(epochs=10)
    apply_patch(ctx, "train.epochs", {"times": 1.5})
    assert ctx.train.epochs == 15 and isinstance(ctx.train.epochs, int)


def test_apply_patch_on_optimizer_and_model_entries():
    ctx = make_ctx()
    optimizer = ctx.optimizers["main"]
    patch = apply_patch(ctx, "optimizers.main.lr", {"times": 0.1}, by="decay")
    assert patch.value == pytest.approx(1e-4) and optimizer.calls == [("lr", pytest.approx(1e-4))]
    apply_patch(ctx, "optimizers.main.weight_decay", 0.01)
    assert optimizer.calls[-1] == ("weight_decay", 0.01)
    patch = apply_patch(ctx, "optimizers.main.minimizes", "@losses.huber")
    assert optimizer.minimizes is ctx.losses["huber"] and patch.value is ctx.losses["huber"]
    assert len(optimizer.calls) == 2
    model = ctx.models["net"]
    apply_patch(ctx, "models.net.freeze", False)
    apply_patch(ctx, "models.net.freeze", ["backbone.*"])
    assert model.frozen == [False, ["backbone.*"]] and model.freeze == ["backbone.*"]
    apply_patch(ctx, "models.net.ema.decay", 0.9)
    assert model.ema.decay == 0.9


def test_apply_patch_errors_name_the_holder_and_key():
    ctx = make_ctx()
    with pytest.raises(RunError, match="Holder has no attribute 'nope'"):
        apply_patch(ctx, "losses.huber.nope", 1)
    with pytest.raises(RunError, match="no loss named 'ghost'"):
        apply_patch(ctx, "losses.ghost.delta", 1)
    with pytest.raises(RunError, match="unknown path root"):
        apply_patch(ctx, "weird.x.y", 1)
    with pytest.raises(RunError):
        apply_patch(ctx, "losses.huber", 1)
    with pytest.raises(RunError):
        apply_patch(ctx, "optimizers.main", 1)
    with pytest.raises(RunError, match="no index of a list"):
        apply_patch(ctx, "losses.total.constraints[3].epsilon", 1)
    with pytest.raises(RunError, match="no index of a list"):
        apply_patch(ctx, "losses.total.constraints.epsilon", 1)
    with pytest.raises(RunError, match="has no attribute 'total'"):
        apply_patch(ctx, "train.updates.total", 5)
    ctx.train = TrainSettings(updates={"total": 10})
    with pytest.raises(RunError, match="'turn' is not a key"):
        apply_patch(ctx, "train.updates.turn.x", 5)
    apply_patch(ctx, "train.updates.turn", 5)
    assert ctx.train.updates == {"total": 10, "turn": 5}
    with pytest.raises(RunError):
        apply_patch(ctx, "rules.ghost.when.at", 5)
    assert ctx.losses["huber"].delta == 1.0


def test_apply_patch_updates_plan_and_writes_events(tmp_path):
    ctx = make_ctx(record=RecordWriter(tmp_path), plan=make_plan())
    apply_patch(ctx, "optimizers.main.lr", {"times": 0.1}, by="decay", turn=2)
    apply_patch(ctx, "optimizers.main.minimizes", "@losses.huber", by="to_huber", turn=3)
    apply_patch(ctx, "losses.total.constraints[0].epsilon", 0.02, by="tighten", turn=4)
    apply_patch(ctx, "models.net.freeze", False, by="thaw", turn=5)
    apply_patch(ctx, "train.epochs", {"plus": 5}, by="longer", turn=6)
    apply_patch(ctx, "train.step.accumulate", 4, by="bigger", turn=7)
    apply_patch(ctx, "losses.huber.delta", {"ref": "models.net", "attr": "ema"}, by="odd", turn=8)
    nodes = ctx.plan["nodes"]
    assert nodes["optimizers.main"]["params"]["lr"] == pytest.approx(1e-4)
    assert nodes["optimizers.main"]["keys"]["minimizes"] == {"ref": "losses.huber"}
    assert nodes["losses.total"]["params"]["constraints"][0]["epsilon"] == 0.02
    assert nodes["models.net"]["keys"]["freeze"] is False
    assert nodes["train.step"]["keys"]["accumulate"] == 4
    assert nodes["losses.huber"]["params"]["delta"] == {"ref": "models.net", "attr": "ema"}
    assert ctx.plan["train"]["epochs"] == 15
    events = events_in(tmp_path)
    assert [event["kind"] for event in events] == ["patch"] * 7
    assert [event["turn"] for event in events] == [2, 3, 4, 5, 6, 7, 8]
    assert events[0]["by"] == "decay" and events[0]["asked"] == {"times": 0.1}
    assert events[0]["value"] == pytest.approx(1e-4)
    assert events[1] == {**events[1], "path": "optimizers.main.minimizes", "value": {"ref": "losses.huber"},
                         "asked": "@losses.huber", "by": "to_huber"}
    assert events[4]["value"] == 15 and events[4]["asked"] == {"plus": 5}
    assert all("at" in event for event in events)


def test_apply_patch_without_record_or_plan_writes_nothing(tmp_path):
    ctx = make_ctx()
    apply_patch(ctx, "losses.huber.delta", 2.0)
    assert ctx.plan is None and events_in(tmp_path) == []


def test_rule_fires_once_and_applies_its_set(caplog):
    ctx = make_ctx()
    ctx.turn = 5
    trigger = FakeTrigger(answer=False)
    rule = Rule(RuleSpec("to_huber", set={"optimizers.main.minimizes": "@losses.huber", "losses.huber.delta": 2.0}),
                when=trigger)
    assert rule.fire(ctx) is False
    assert ctx.optimizers["main"].minimizes is ctx.losses["mse"] and ctx.fired == []
    trigger.answer = True
    with caplog.at_level(logging.INFO, logger="kalfa.run"):
        assert rule.fire(ctx) is True
    assert "to_huber fired at turn 5" in caplog.text
    assert ctx.optimizers["main"].minimizes is ctx.losses["huber"] and ctx.losses["huber"].delta == 2.0
    assert rule.fired_turn == 5 and rule.done is True and ctx.fired == ["to_huber"]
    assert trigger.resets == 0
    ctx.turn = 6
    assert rule.fire(ctx) is False
    assert ctx.fired == ["to_huber"]


def test_rule_without_once_fires_again_and_resets_its_trigger():
    ctx = make_ctx()
    trigger = FakeTrigger()
    action = FakeAction()
    rule = Rule(RuleSpec("samples", once=False), when=trigger, run=action)
    ctx.turn = 5
    assert rule.fire(ctx) is True
    ctx.turn = 6
    assert rule.fire(ctx) is True
    assert action.calls == 2 and trigger.resets == 2 and rule.done is False and rule.fired_turn == 6
    assert ctx.fired == ["samples", "samples"]


def test_rule_after_waits_for_an_earlier_turn_and_stop_names_the_trigger():
    ctx = make_ctx()
    first = Rule(RuleSpec("first", set={"losses.huber.delta": 3.0}))
    second = Rule(RuleSpec("second", after="first", stop=True))
    third = Rule(RuleSpec("third", after="first", stop=True), when=FakeTrigger())
    ctx.rules = [first, second, third]
    ctx.turn = 2
    assert second.ready(ctx) is False and second.fire(ctx) is False
    assert first.fire(ctx) is True
    assert second.ready(ctx) is False and second.fire(ctx) is False
    ctx.turn = 3
    assert second.ready(ctx) is True and second.fire(ctx) is True
    assert ctx.stopping == "second: rule"
    assert third.fire(ctx) is True
    assert ctx.stopping == "third: fake trigger"
    assert ctx.fired == ["first", "second", "third"]
    assert ctx.rules_by_name == {"first": first, "second": second, "third": third}
    orphan = Rule(RuleSpec("orphan", after="ghost", stop=True))
    with pytest.raises(RunError, match="no rule"):
        orphan.fire(ctx)


def test_rule_without_when_fires_when_ready():
    ctx = make_ctx()
    ctx.turn = 1
    rule = Rule(RuleSpec("now", set={"losses.huber.delta": 9.0}))
    assert rule.fire(ctx) is True and ctx.losses["huber"].delta == 9.0
    assert rule.fire(ctx) is False


def test_rule_state_dict_round_trip():
    rule = Rule(RuleSpec("a", stop=True))
    rule.fired_turn, rule.done = 4, True
    fresh = Rule(RuleSpec("a", stop=True))
    fresh.load_state_dict(rule.state_dict())
    assert fresh.fired_turn == 4 and fresh.done is True


def test_replay_reapplies_resolved_values_without_new_events(tmp_path):
    source = make_ctx(record=RecordWriter(tmp_path / "source"))
    apply_patch(source, "optimizers.main.lr", {"times": 0.1}, by="decay", turn=1)
    apply_patch(source, "optimizers.main.minimizes", "@losses.huber", by="to_huber", turn=2)
    apply_patch(source, "losses.huber.delta", {"plus": 1}, by="later", turn=3)
    source.record.event("stop", turn=3, reason="done")
    events = events_in(tmp_path / "source")
    fresh = make_ctx(record=RecordWriter(tmp_path / "fresh"))
    assert replay(fresh, events, upto_turn=2) == 2
    assert fresh.optimizers["main"].params["lr"] == pytest.approx(1e-4)
    assert fresh.optimizers["main"].minimizes is fresh.losses["huber"]
    assert fresh.losses["huber"].delta == 1.0
    assert events_in(tmp_path / "fresh") == []
    assert fresh.record is not None
    assert replay(fresh, events, upto_turn=3) == 3
    assert fresh.losses["huber"].delta == 2.0
    assert replay(fresh, [{"kind": "turn", "turn": 1}], upto_turn=5) == 0
