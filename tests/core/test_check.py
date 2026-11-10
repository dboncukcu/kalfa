import pytest
import torch
from torch import nn

from kalfa.core import aliases, build, plan, spec
from kalfa.core.build import Node, Ref, Tree
from kalfa.core.check import Checked, check, validate
from kalfa.core.rules import parse_rules
from kalfa.core.errors import error, warning
from kalfa.core.protocols import ContextMetric, Objective, Trigger
from kalfa.core.rules import RuleSpec
from kalfa.core.spec import Config


class FakeTable:
    inputs = ("x",)
    exposes = ("n_features", "feature_names", "target_names", "n_classes", "class_weights", "classes")

    def __init__(self, file, split=None, features=None, targets=None, filter=None, drop=None, spectators=None,
                 feed=None, batch=128, stream=False):
        self.file = file
        self.targets = dict(targets or {})


class FakeFolder:
    inputs = ("image",)
    exposes = ("n_classes",)

    def __init__(self, folder, split=None, batch=32):
        self.folder = folder


class Net(nn.Module):
    def __init__(self, hidden, out):
        super().__init__()
        self.layer = nn.Linear(1, out)

    def forward(self, x):
        return self.layer(x)


class Rmse:
    def __init__(self):
        self.total = 0.0

    def update(self, pred, target):
        self.total += float(((pred - target) ** 2).mean())

    def compute(self):
        return self.total

    def reset(self):
        self.total = 0.0


class Elbo(Objective):
    def __init__(self, recon, w_rec=1.0):
        super().__init__()
        self.recon = recon
        self.w_rec = float(w_rec)

    def forward(self, outputs, batch, ctx):
        return self.recon(outputs, batch, ctx) * self.w_rec


class Watching(ContextMetric):
    def __init__(self, series):
        self.series = series


class LogCosh:
    def __call__(self, pred, target):
        return torch.log(torch.cosh(pred - target)).mean()


class AfterEpoch(Trigger):
    def __init__(self, at):
        self.at = int(at)

    def fires(self, ctx):
        return ctx.turn >= self.at


class CustomStep:
    def __init__(self, order=None):
        self.order = order


MUTABLE = {"optimizers.main.lr", "optimizers.main.weight_decay", "optimizers.main.minimizes", "models.net.freeze",
           "losses.huber.delta", "losses.mse.on", "losses.mse.every", "metrics.rmse.every", "train.epochs",
           "train.updates.total", "rules.to_huber.when.at"}


def node(path, slot, use, target, params=None, keys=None):
    return Node(path, slot, use, target, dict(params or {}), dict(keys or {}))


def make_tree(nodes, settings=None, rules=None, data=None):
    config = Config(data=dict(data or {}), params={}, plugins=[], layers=[], directories=[], overrides={})
    top = {key: value for key, value in (data or {}).items() if key in ("seed", "device", "record")}
    top["train"] = {"epochs": 3} if settings is None else settings
    problems = []
    rules = parse_rules(rules, problems) if isinstance(rules, dict) else list(rules or [])
    return Tree(config, {item.path: item for item in nodes}, [item.path for item in nodes], top, rules, problems)


def replace(nodes, path, item):
    nodes[[existing.path for existing in nodes].index(path)] = item


def housing_nodes(models=("net",), targets=("price",), pred=None, target=None, data=True):
    nodes = []
    if data:
        nodes.append(node("data", "data", "table", FakeTable, {
            "file": "housing.parquet", "split": [0.7, 0.15, 0.15], "features": {"x*": "standard_scaler"},
            "targets": {name: "standard_scaler" for name in targets}, "batch": 128}))
    for name in models:
        nodes.append(node(f"models.{name}", "model", "mlp", Net, {"hidden": [8], "out": 1}))
    keys = {key: value for key, value in (("pred", pred), ("target", target)) if value is not None}
    nodes.append(node("losses.mse", "loss", "mse", nn.MSELoss, {}, dict(keys)))
    nodes.append(node("losses.huber", "loss", "torch.nn.HuberLoss", nn.HuberLoss, {"delta": 1.0}, dict(keys)))
    nodes.append(node("metrics.rmse", "metric", "rmse", Rmse, {}, dict(keys)))
    nodes.append(node("optimizers.main", "optimizer", "adam", torch.optim.Adam, {"lr": 1e-3},
                      {"over": "@models.net", "minimizes": "@losses.mse"}))
    nodes.append(node("rules.to_huber.when", "trigger", "after_epoch", AfterEpoch, {"at": 2}))
    return nodes


def codes(problems):
    return [problem.code for problem in problems]


def problem(problems, code):
    found = [item for item in problems if item.code == code]
    assert found, f"no {code} in {[str(item) for item in problems]}"
    return found[0]


@pytest.fixture
def fake_mutable(monkeypatch):
    monkeypatch.setattr(build, "mutable_paths", lambda tree: set(MUTABLE))


def test_validate_accepts_the_housing_setup(fake_mutable):
    rules = [RuleSpec("to_huber", when={"use": "after_epoch", "at": 2},
                      set={"optimizers.main.minimizes": "@losses.huber", "optimizers.main.lr": {"times": 0.5},
                           "losses.huber.delta": 0.5, "models.net.freeze": False, "train.epochs": {"plus": 1},
                           "rules.to_huber.when.at": 3}),
             RuleSpec("stop", after="to_huber", when={"use": "plateau", "watch": "val/rmse", "patience": 6}, stop=True)]
    tree = make_tree(housing_nodes(), {"epochs": 100, "checkpoint": {"watch": "val/rmse"}, "report": "best",
                                       "evaluate": ["valid", "test"], "predicts": "@models.net",
                                       "track": ["losses.huber.delta", "train.epochs", "losses.mse.lmbda[1]"]},
                     rules, {"device": "cpu"})
    assert validate(tree) == []


def test_validate_not_mutable(fake_mutable):
    rules = [RuleSpec("widen", set={"models.net.hidden": [16]}),
             RuleSpec("retarget", set={"losses.mse.target": "price"})]
    problems = validate(make_tree(housing_nodes(target="price"), rules=rules))
    assert codes(problems) == ["not_mutable", "not_mutable"]
    first = problem(problems, "not_mutable")
    assert first.path == "rules.widen.set" and "models.net.hidden" in first.message
    assert first.hint == "mutable under models.net: models.net.freeze"


def test_validate_unknown_path(fake_mutable):
    rules = [RuleSpec("a", set={"optimizers.main.lrr": 1}), RuleSpec("b", set={"optimizers.nope.lr": 1}),
             RuleSpec("c", set={"losses.total.constraints[0].epsilon": 1}),
             RuleSpec("d", set={"train.updates.turn": 1})]
    problems = validate(make_tree(housing_nodes(), rules=rules))
    assert codes(problems) == ["unknown_path"] * 4
    assert problems[0].hint == ("mutable under optimizers.main: optimizers.main.lr, optimizers.main.minimizes, "
                                "optimizers.main.weight_decay")
    assert problems[1].hint == "the optimizers are main"
    assert problems[2].hint == "the losses are huber, mse"
    assert problems[3].hint == "mutable under train: train.epochs, train.updates.total"


def test_validate_rule_references_and_cycles(fake_mutable):
    rules = [RuleSpec("a", after="b", set={"optimizers.main.minimizes": "@losses.ghost"}),
             RuleSpec("b", after="a", set={"optimizers.main.minimizes": "@losses.huber"}),
             RuleSpec("c", after="a", stop=True)]
    problems = validate(make_tree(housing_nodes(), rules=rules))
    assert codes(problems) == ["rule_ref", "rule_after"]
    assert "a, b" in problems[1].message and problems[1].path == "rules.a.after"


def test_validate_train_settings(fake_mutable):
    nodes = housing_nodes()
    both = validate(make_tree(nodes, {"epochs": 3, "updates": {"total": 10, "turn": 5}}))
    assert codes(both) == ["train_unit"]
    neither = validate(make_tree(nodes, {}))
    assert codes(neither) == ["train_unit"]
    assert codes(validate(make_tree(nodes, {"updates": {"total": 10, "turn": 5}}))) == []
    assert codes(validate(make_tree(nodes, {"updates": {"total": 0}}))) == ["train_unit"]
    assert codes(validate(make_tree(nodes, {"updates": {"total": 10, "step": 5}}))) == ["train_key"]
    assert codes(validate(make_tree(nodes, {"epochs": -1}))) == ["train_unit"]
    assert codes(validate(make_tree(nodes, {"epochs": True}))) == ["train_unit"]
    assert codes(validate(make_tree(nodes, {"epochs": 3, "report": "best"}))) == ["report_best"]
    assert codes(validate(make_tree(nodes, {"epochs": 3, "report": "best", "checkpoint": {"mode": "min"}}))) == [
        "report_best"]
    assert codes(validate(make_tree(nodes, {"epochs": 3, "report": "worst"}))) == ["train_report"]
    assert codes(validate(make_tree(nodes, {"epochs": 3, "epoch": 3}))) == ["train_key"]
    assert codes(validate(make_tree(nodes, {"epochs": 3, "evaluate": "valid"}))) == ["evaluate_set"]
    assert codes(validate(make_tree(nodes, {"epochs": 3, "evaluate": ["valid", "holdout"]}))) == ["evaluate_set"]
    assert codes(validate(make_tree(nodes, {"epochs": 3, "evaluate": ["test", "test"]}))) == ["evaluate_set"]
    assert codes(validate(make_tree(nodes, {"epochs": 3, "predicts": "@models.ghost"}))) == ["predicts_ref"]
    assert codes(validate(make_tree(nodes, {"epochs": 3, "predicts": "net"}))) == ["predicts_ref"]
    assert codes(validate(make_tree(nodes, {"epochs": 3, "track": "a.b"}))) == ["train_track"]
    checkpoint = {"watch": "val/rmse", "mode": "avg", "every": 0, "last": "no", "keep": 3}
    found = codes(validate(make_tree(nodes, {"epochs": 3, "checkpoint": checkpoint})))
    assert found.count("checkpoint_value") == 3 and found.count("checkpoint_key") == 1
    assert codes(validate(make_tree(nodes, {"epochs": 3, "checkpoint": {"watch": 5}}))) == ["checkpoint_watch"]
    assert codes(validate(make_tree(nodes, "epochs"))) == ["train_unit"]


def test_validate_evaluate_names_are_free_for_other_data_kinds(fake_mutable):
    nodes = housing_nodes(data=False)
    nodes.insert(0, node("data", "data", "image_folder", FakeFolder, {"folder": "images", "batch": 32}))
    assert validate(make_tree(nodes, {"epochs": 3, "evaluate": ["holdout"]})) == []


def test_validate_optimizer_minimizes(fake_mutable):
    nodes = housing_nodes()
    nodes[-2].keys["minimizes"] = "@metrics.rmse"
    assert codes(validate(make_tree(nodes))) == ["minimizes_ref"]
    nodes[-2].keys["minimizes"] = "@losses.huber.delta"
    assert codes(validate(make_tree(nodes))) == ["minimizes_ref"]
    nodes[-2].keys["minimizes"] = Ref("@losses.ghost", "losses.ghost", None, "minimizes")
    found = validate(make_tree(nodes))
    assert codes(found) == ["minimizes_ref"] and "@losses.ghost" in found[0].message
    nodes[-2].keys["minimizes"] = Ref("@losses.huber", "losses.huber", None, "minimizes")
    assert validate(make_tree(nodes)) == []
    del nodes[-2].keys["minimizes"]
    missing = validate(make_tree(nodes))
    assert codes(missing) == ["minimizes_required"] and missing[0].severity == "error"
    nodes.append(node("train.step", "step", "mysteps.Gan", CustomStep, {}, {"order": ["main"]}))
    softer = validate(make_tree(nodes))
    assert codes(softer) == ["minimizes_required"] and softer[0].severity == "warning"
    nodes[-1] = node("train.step", "step", "alternating", CustomStep, {}, {"order": ["main"]})
    assert validate(make_tree(nodes))[0].severity == "error"


def test_validate_pred_required(fake_mutable):
    nodes = housing_nodes(models=("a", "b"))
    problems = validate(make_tree(nodes))
    assert codes(problems) == ["pred_required"] * 3
    assert [item.path for item in problems] == ["losses.mse", "losses.huber", "metrics.rmse"]
    assert problems[0].hint == "the outputs are a, b"
    assert validate(make_tree(nodes, {"epochs": 3, "predicts": "@models.a"})) == []
    assert validate(make_tree(housing_nodes(models=("a", "b"), pred="a"))) == []
    assert codes(validate(make_tree(housing_nodes(models=())))) == ["pred_required"] * 3
    objective = housing_nodes(models=("a", "b"))
    replace(objective, "losses.mse", node("losses.mse", "loss", "mymodels.Elbo", Elbo,
                                           {"recon": "@losses.huber", "w_rec": 1.0}))
    replace(objective, "metrics.rmse", node("metrics.rmse", "metric", "mymodels.Watching", Watching,
                                             {"series": "val/mse"}))
    problems = validate(make_tree(objective))
    assert codes(problems) == ["pred_required"] and problems[0].path == "losses.huber"
    broken = housing_nodes(models=("a", "b"))
    replace(broken, "losses.mse", node("losses.mse", "loss", "mylegos.gone", None))
    assert codes(validate(make_tree(broken))) == ["pred_required"] * 2


def test_validate_unknown_output_is_a_warning(fake_mutable):
    nodes = housing_nodes(pred="y_hat")
    problems = validate(make_tree(nodes))
    assert codes(problems) == ["unknown_output"] * 3
    assert all(item.severity == "warning" for item in problems)
    assert problems[0].path == "losses.mse.pred" and problems[0].hint == "the outputs are net"
    nodes = housing_nodes(pred="x_hat")
    nodes[1].keys["outputs"] = ["x_hat", "mu", "logvar"]
    assert validate(make_tree(nodes)) == []


def test_validate_target_required(fake_mutable):
    problems = validate(make_tree(housing_nodes(targets=("price", "rooms"))))
    assert codes(problems) == ["target_required"] * 3
    assert problems[0].hint == "the targets are price, rooms"
    assert validate(make_tree(housing_nodes(targets=("price", "rooms"), target="price"))) == []
    none = validate(make_tree(housing_nodes(targets=())))
    assert codes(none) == ["target_required"] * 3 and none[0].hint == "the data declares no targets"
    nodes = housing_nodes(data=False)
    nodes.insert(0, node("data", "data", "image_folder", FakeFolder, {"folder": "images", "batch": 32}))
    assert validate(make_tree(nodes)) == []
    assert validate(make_tree(housing_nodes(data=False))) == []


def test_validate_device(fake_mutable):
    nodes = housing_nodes()
    for good in ("auto", "cpu", "cuda", "cuda:1", "mps"):
        assert validate(make_tree(nodes, data={"device": good})) == []
    assert codes(validate(make_tree(nodes, data={"device": "gpu"}))) == ["device_value"]
    assert codes(validate(make_tree(nodes, data={"device": 0}))) == ["device_value"]


def test_validate_warns_about_unknown_series(fake_mutable):
    nodes = housing_nodes()
    nodes[-1].params["watch"] = "val/accuracy"
    problems = validate(make_tree(nodes, {"epochs": 3, "checkpoint": {"watch": "val/nope"}}))
    assert codes(problems) == ["unknown_series", "unknown_series"]
    assert all(item.severity == "warning" for item in problems)
    assert problems[0].path == "train.checkpoint.watch" and problems[1].path == "rules.to_huber.when.watch"
    nodes[-1].params["watch"] = "train/huber"
    fine = {"epochs": 3, "checkpoint": {"watch": "holdout/rmse"}, "evaluate": ["holdout"]}
    nodes[0] = node("data", "data", "image_folder", FakeFolder, {"folder": "images"})
    assert validate(make_tree(nodes, fine)) == []
    assert codes(validate(make_tree(nodes, {"epochs": 3, "checkpoint": {"watch": "rmse"}}))) == ["unknown_series"]


def test_validate_reads_raw_rules_and_tolerates_missing_sections(fake_mutable):
    assert validate(make_tree([], {"epochs": 2})) == []
    raw = {"widen": {"set": {"models.net.hidden": [16]}}, "late": {"after": "ghost", "stop": True}}
    problems = validate(make_tree(housing_nodes(), rules=raw))
    assert codes(problems) == ["not_mutable"]
    specs = [RuleSpec("late", after="ghost", stop=True)]
    assert validate(make_tree(housing_nodes(), rules=specs)) == []


@pytest.fixture
def fake_pipeline(monkeypatch, fake_mutable):
    state = {"dirs": [], "plugins": None, "planned": 0}

    def load(paths=(), sets=(), params=(), mappings=()):
        state["loaded"] = (tuple(paths), tuple(sets), tuple(params), list(mappings))
        return state["config"]

    def plan_of(tree, version=None):
        state["planned"] += 1
        return {"format": 1, "kalfa": version, "nodes": sorted(tree.nodes)}

    monkeypatch.setattr(spec, "load", load)
    monkeypatch.setattr(aliases, "add_search_dir", lambda directory: state["dirs"].append(str(directory)))
    monkeypatch.setattr(aliases, "import_plugins", lambda names, problems: state.__setitem__("plugins", list(names)))
    monkeypatch.setattr(build, "build", lambda config: state["tree"])
    monkeypatch.setattr(plan, "plan_of", plan_of)
    return state


def housing_config(rules=None):
    return {"data": {"use": "table", "file": "housing.parquet", "targets": {"price": "standard_scaler"}},
            "models": {"net": {"use": "mlp", "hidden": [8], "out": 1}}, "losses": {"mse": "mse", "huber": "huber"},
            "metrics": {"rmse": "rmse"},
            "optimizers": {"main": {"use": "adam", "lr": 1e-3, "over": "@models.net", "minimizes": "@losses.mse"}},
            "train": {"epochs": 3}, "rules": rules or {}}


def test_check_wires_load_build_rules_validate_and_plan(fake_pipeline):
    rules = {"to_huber": {"when": {"use": "after_epoch", "at": 2},
                          "set": {"optimizers.main.minimizes": "@losses.huber"}}}
    data = housing_config(rules)
    fake_pipeline["config"] = Config(data=data, params={}, plugins=["mylegos"], layers=[], directories=["/tmp/configs"],
                                     overrides={})
    fake_pipeline["tree"] = make_tree(housing_nodes(), data["train"], rules=data["rules"])
    checked = check(paths=["a.yaml"], sets=["train.epochs=3"], params=["lr=0.1"], mappings=[{"x": 1}])
    assert isinstance(checked, Checked) and checked.ok and checked.problems == [] and checked.text() == "ok"
    assert checked.plan["format"] == 1 and checked.plan["kalfa"].startswith("3.") and "data" in checked.plan["nodes"]
    assert checked.tree is fake_pipeline["tree"] and checked.config is fake_pipeline["config"]
    assert [rule.name for rule in checked.tree.rules] == ["to_huber"]
    assert fake_pipeline["loaded"] == (("a.yaml",), ("train.epochs=3",), ("lr=0.1",), [{"x": 1}])
    assert fake_pipeline["dirs"][0] == "/tmp/configs" and len(fake_pipeline["dirs"]) == 2
    assert fake_pipeline["plugins"] == ["mylegos"] and fake_pipeline["planned"] == 1


def test_check_reports_rule_problems_of_the_tree(fake_pipeline):
    data = housing_config({"widen": {"set": {"models.net.hidden": [16]}}, "orphan": {"after": "ghost", "stop": True}})
    fake_pipeline["config"] = Config(data=data, params={}, plugins=[], layers=[], directories=[], overrides={})
    fake_pipeline["tree"] = make_tree(housing_nodes(), data["train"], rules=data["rules"])
    checked = check(mappings=[data])
    assert codes(checked.problems) == ["rule_after", "not_mutable"]
    assert not checked.ok and checked.plan is None and fake_pipeline["planned"] == 0
    assert "error not_mutable: rules.widen.set: models.net.hidden is fixed" in checked.text()
def test_check_stops_at_load_errors_and_merges_build_problems(fake_pipeline):
    data = housing_config()
    broken = error("unknown_param", "no param named lr", "optimizers.main.lr")
    fake_pipeline["config"] = Config(data=data, params={}, plugins=[], layers=[], directories=[], overrides={},
                                     problems=[broken])
    fake_pipeline["tree"] = make_tree(housing_nodes(), data["train"])
    checked = check(mappings=[data])
    assert checked.problems == [broken] and checked.tree is None and checked.plan is None
    assert fake_pipeline["plugins"] is None
    assert checked.text() == str(broken)
    soft = warning("signature_open", "signature of mylegos.Thing could not be read", "models.net")
    fake_pipeline["config"] = Config(data=data, params={}, plugins=[], layers=[], directories=[], overrides={},
                                     problems=[soft])
    tree = make_tree(housing_nodes(), data["train"])
    tree.problems = [soft, error("unknown_key", "no such key", "models.net.width")]
    fake_pipeline["tree"] = tree
    checked = check(mappings=[data])
    assert codes(checked.problems) == ["signature_open", "unknown_key"]
    assert checked.warnings == [soft] and len(checked.errors) == 1 and checked.plan is None and checked.tree is tree


@pytest.fixture(scope="module")
def fakes():
    aliases.register("c_table", FakeTable, slot="data", doc="table stand-in for the check tests", plugin=True)
    aliases.register("c_net", Net, slot="model", doc="tiny model for the check tests", plugin=True)
    aliases.register("c_rmse", Rmse, slot="metric", doc="metric stand-in for the check tests", plugin=True)
    aliases.register("c_log_cosh", LogCosh, slot="loss", doc="criterion stand-in for the check tests", plugin=True)
    aliases.register("c_after_epoch", AfterEpoch, slot="trigger", doc="trigger for the check tests", plugin=True)
    yield
    for name in ("c_table", "c_net", "c_rmse", "c_log_cosh", "c_after_epoch"):
        aliases.ALIASES.pop(name, None)


def checked(mapping):
    return check(mappings=[mapping])


def config_01():
    return {
        "params": {"epochs": 100, "lr": 1.0e-3, "seed": 7},
        "seed": "$seed$",
        "data": {"use": "c_table", "file": "housing.parquet", "split": [0.7, 0.15, 0.15],
                 "features": {"x*": "standard_scaler"}, "targets": {"price": "standard_scaler"}, "batch": 128},
        "models": {"net": {"use": "mlp", "hidden": [128, 128], "out": 1}},
        "losses": {"mse": "mse", "huber": {"use": "torch.nn.HuberLoss", "delta": 1.0}, "mae": "torch.nn.L1Loss",
                   "log_cosh": "c_log_cosh"},
        "metrics": {"rmse": "rmse", "mae": "mae"},
        "optimizers": {"main": {"use": "adam", "lr": "$lr$", "over": "@models.net", "minimizes": "@losses.mse"}},
        "train": {"epochs": "$epochs$", "checkpoint": {"watch": "val/rmse"}, "report": "best"},
        "rules": {
            "to_huber": {"when": {"use": "after_epoch", "at": 50},
                         "set": {"optimizers.main.minimizes": "@losses.huber"}},
            "to_mae": {"after": "to_huber", "when": {"use": "below", "watch": "train/huber", "value": 0.01},
                       "set": {"optimizers.main.minimizes": "@losses.mae"}},
            "stop": {"when": {"use": "plateau", "watch": "val/rmse", "patience": 6}, "stop": True},
        },
        "plots": {"loss_curve": "loss_curve", "pred_vs_true": "pred_vs_true"},
        "record": "runs/housing_$datetime$",
    }


def minimal(**changes):
    config = {
        "seed": 1,
        "data": {"use": "c_table", "file": "housing.parquet", "split": [0.7, 0.15, 0.15],
                 "features": {"x*": "standard_scaler"}, "targets": {"price": "standard_scaler"}, "batch": 32},
        "models": {"net": {"use": "c_net", "hidden": [8], "out": 1}},
        "losses": {"mse": "torch.nn.MSELoss", "huber": {"use": "torch.nn.HuberLoss", "delta": 1.0}},
        "metrics": {"rmse": "c_rmse"},
        "optimizers": {"main": {"use": "torch.optim.Adam", "lr": 1e-3, "over": "@models.net",
                                "minimizes": "@losses.mse"}},
        "train": {"epochs": 3},
        "rules": {"to_huber": {"when": {"use": "c_after_epoch", "at": 2},
                               "set": {"optimizers.main.minimizes": "@losses.huber"}}},
    }
    config.update(changes)
    return config


def test_check_accepts_the_01_config(fakes):
    result = checked(config_01())
    assert result.errors == [], result.text()
    assert result.ok and result.plan is not None and result.tree is not None
    assert [rule.name for rule in result.tree.rules] == ["to_huber", "to_mae", "stop"]
    assert {"rules.to_huber.when", "rules.to_mae.when", "rules.stop.when"} <= set(result.tree.nodes)


def test_check_accepts_the_minimal_config(fakes):
    result = checked(minimal())
    assert result.errors == [], result.text()
    assert result.plan is not None


def test_check_reports_not_mutable(fakes):
    rules = {"widen": {"when": {"use": "c_after_epoch", "at": 2}, "set": {"models.net.hidden": [16]}}}
    result = checked(minimal(rules=rules))
    assert "not_mutable" in codes(result.problems), result.text()
    assert not result.ok and result.plan is None


def test_check_reports_unknown_path(fakes):
    rules = {"typo": {"when": {"use": "c_after_epoch", "at": 2}, "set": {"optimizers.main.lrr": 0.1}}}
    result = checked(minimal(rules=rules))
    assert "unknown_path" in codes(result.problems), result.text()
    assert not result.ok


def test_check_reports_pred_required(fakes):
    models = {"a": {"use": "c_net", "hidden": [8], "out": 1}, "b": {"use": "c_net", "hidden": [8], "out": 1}}
    optimizers = {"main": {"use": "torch.optim.Adam", "lr": 1e-3, "over": ["@models.a", "@models.b"],
                           "minimizes": "@losses.mse"}}
    result = checked(minimal(models=models, optimizers=optimizers))
    assert "pred_required" in codes(result.problems), result.text()
    assert not result.ok


def test_check_reports_target_required(fakes):
    data = minimal()["data"]
    data["targets"] = {"price": "standard_scaler", "rooms": "standard_scaler"}
    result = checked(minimal(data=data))
    assert "target_required" in codes(result.problems), result.text()
    assert not result.ok


def test_check_reports_rule_after(fakes):
    rules = {"late": {"after": "ghost", "stop": True}}
    result = checked(minimal(rules=rules))
    assert "rule_after" in codes(result.problems), result.text()
    assert not result.ok
