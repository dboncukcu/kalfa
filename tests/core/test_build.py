from types import SimpleNamespace

import pytest
import torch
from torch import nn

from kalfa.core import wrap
from kalfa.core.aliases import ALIASES, Entry
from kalfa.core.build import Node, Ref, Tree, build, instantiate, mutable_paths, prepare_data, resolve_refs
from kalfa.core.context import Context
from kalfa.core.errors import BuildError, KalfaError, ProblemsError
from kalfa.core.protocols import Action, Objective, Step, Trigger


class FakeSplit:
    def __init__(self, folds, fold, seed=None):
        self.folds = int(folds)
        self.fold = int(fold)
        self.seed = seed


class FakeData:
    inputs = ("x",)
    exposes = ("n_features", "n_classes", "class_weights")

    def __init__(self, file, split=None, features=None, targets=None, batch=128):
        self.file = file
        self.split = split
        self.features = features
        self.targets = targets
        self.batch = batch
        self.n_features = 3
        self.n_classes = 2
        self.class_weights = None
        self.public = "open"
        self._secret = "hidden"
        self.fitted = False
        self.fit_ctx = None

    def fit(self, ctx):
        self.fitted = True
        self.fit_ctx = ctx


class FakeStep(Step):
    def __init__(self, repeat=None, fresh_batch=False):
        self.repeat = dict(repeat or {})
        self.fresh_batch = bool(fresh_batch)
        self.bound = None

    def bind(self, ctx):
        self.bound = ctx


class FakeObjective(Objective):
    def __init__(self, recon, w_rec=1.0, w_kl=1.0):
        super().__init__()
        self.recon = recon
        self.w_rec = w_rec
        self.w_kl = w_kl
        self.bound = None

    def bind(self, ctx):
        self.bound = ctx

    def forward(self, outputs, batch, ctx):
        return torch.zeros(())


class FakeMdmm(Objective):
    def __init__(self, objective, constraints):
        super().__init__()
        self.objective = objective
        self.constraints = list(constraints)
        self.lmbda = nn.Parameter(torch.zeros(len(self.constraints)))

    def forward(self, outputs, batch, ctx):
        return self.lmbda.sum()


def mse():
    return nn.MSELoss()


class FakeMetric:
    def __init__(self):
        self.seen = 0

    def reset(self):
        self.seen = 0

    def update(self, pred, target):
        self.seen += 1

    def compute(self):
        return float(self.seen)


class FakePlateau(Trigger):
    fixed = ("watch",)

    def __init__(self, watch, patience, mode="min", min_delta=0.0):
        self.watch = watch
        self.patience = int(patience)
        self.mode = mode
        self.min_delta = float(min_delta)

    def fires(self, ctx):
        return False


class FakeAfterEpoch(Trigger):
    def __init__(self, at):
        self.at = int(at)

    def fires(self, ctx):
        return ctx.turn >= self.at


class FakeEvery(Trigger):
    def __init__(self, epochs):
        self.epochs = int(epochs)

    def fires(self, ctx):
        return ctx.turn % self.epochs == 0


class FakeWriteSamples(Action):
    def __init__(self, sampler, n=16):
        self.sampler = sampler
        self.n = int(n)

    def __call__(self, ctx):
        return self.sampler.sample(ctx, self.n)


class FakeSampler:
    def __init__(self, model, steps=10):
        self.model = model
        self.steps = int(steps)

    def sample(self, ctx, n):
        return torch.zeros(n)


class FakeWarmup:
    def __init__(self, start, end, steps):
        self.start = float(start)
        self.end = float(end)
        self.steps = int(steps)


class FakeInit:
    def __init__(self, std=0.02):
        self.std = float(std)

    def __call__(self, module):
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.fill_(self.std)


def fake_plot(record, figures, series=None):
    return None


class OpenModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.layer = nn.Linear(2, 1)

    def forward(self, x):
        return self.layer(x)


class ClashModel(nn.Module):
    def __init__(self, inputs=2):
        super().__init__()
        self.layer = nn.Linear(inputs, 1)

    def forward(self, x):
        return self.layer(x)


class Ae(nn.Module):
    def __init__(self, latent=4):
        super().__init__()
        self.encoder = nn.Linear(3, latent)
        self.decoder = nn.Linear(latent, 3)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class Named:
    def __init__(self, name, path="given"):
        self.name = name
        self.path = path

    def fires(self, ctx):
        return False


FAKES = {
    "f_table": (FakeData, "data"),
    "f_kfold": (FakeSplit, "strategy"),
    "supervised": (FakeStep, "step"),
    "f_alternating": (FakeStep, "step"),
    "f_mse": (mse, "loss"),
    "f_elbo": (FakeObjective, "loss"),
    "f_mdmm": (FakeMdmm, "loss"),
    "f_rmse": (FakeMetric, "metric"),
    "f_plateau": (FakePlateau, "trigger"),
    "f_after_epoch": (FakeAfterEpoch, "trigger"),
    "f_every": (FakeEvery, "trigger"),
    "f_write_samples": (FakeWriteSamples, "action"),
    "f_sampler": (FakeSampler, "sampler"),
    "f_warmup": (FakeWarmup, "schedule"),
    "f_init": (FakeInit, "value"),
    "f_plot": (fake_plot, "plot"),
    "f_open": (OpenModel, "model"),
    "f_clash": (ClashModel, "model"),
    "f_ae": (Ae, "model"),
    "f_named": (Named, "trigger"),
    "f_adam": ("torch.optim.Adam", "optimizer"),
}


@pytest.fixture(autouse=True)
def fakes():
    saved = dict(ALIASES)
    for name, (target, slot) in FAKES.items():
        ALIASES[name] = Entry(name, target, slot)
    yield
    ALIASES.clear()
    ALIASES.update(saved)


@pytest.fixture
def wrapped(monkeypatch):
    def wrap_loss(obj, keys, name, path):
        if isinstance(obj, Objective):
            obj.on = tuple(keys.get("on", obj.on))
            return obj
        criterion = wrap.Criterion(obj, pred=keys.get("pred"), target=keys.get("target"),
                                   rescale=keys.get("rescale", False), on=keys.get("on", ("train", "valid", "test")),
                                   every=keys.get("every", 1))
        criterion.name = name
        criterion.path = path
        return criterion

    def wrap_metric(obj, keys, name, path):
        return wrap.MetricEntry(name, obj, pred=keys.get("pred"), target=keys.get("target"),
                                rescale=keys.get("rescale"), on=keys.get("on", ("train", "valid", "test")),
                                every=keys.get("every", 1))

    def build_optimizer(self):
        parameters = []
        seen = set()
        for item in self.over:
            holder = item.of if isinstance(item, wrap.Group) else item
            for parameter in holder.parameters():
                if id(parameter) not in seen:
                    seen.add(id(parameter))
                    parameters.append(parameter)
        self.real = self.target(parameters, **self.params)

    def set_freeze(self, value):
        self.freeze = value
        self.asked = value

    monkeypatch.setattr(wrap, "wrap_loss", wrap_loss)
    monkeypatch.setattr(wrap, "wrap_metric", wrap_metric)
    monkeypatch.setattr(wrap.OptimizerEntry, "build", build_optimizer)
    monkeypatch.setattr(wrap.ModelEntry, "set_freeze", set_freeze)


def housing():
    return {
        "seed": 7,
        "data": {"use": "f_table", "file": "housing.parquet", "split": [0.7, 0.15, 0.15],
                 "features": {"x*": "standard_scaler"}, "targets": {"price": "standard_scaler"}, "batch": 128},
        "models": {"net": {"use": "torch.nn.Linear", "in_features": "@data.n_features", "out_features": 1}},
        "losses": {"mse": "f_mse", "huber": {"use": "torch.nn.HuberLoss", "delta": 1.0}, "mae": "torch.nn.L1Loss"},
        "metrics": {"rmse": {"use": "f_rmse", "on": ["valid", "test"]}, "mae": "f_rmse"},
        "optimizers": {"main": {"use": "f_adam", "lr": 1e-3, "over": "@models.net", "minimizes": "@losses.mse"}},
        "train": {"epochs": 3, "checkpoint": {"watch": "val/rmse"}, "report": "best",
                  "track": ["losses.huber.delta"]},
        "rules": {
            "to_huber": {"when": {"use": "f_after_epoch", "at": 2},
                         "set": {"optimizers.main.minimizes": "@losses.huber"}},
            "to_mae": {"after": "to_huber", "when": {"use": "f_plateau", "watch": "train/huber", "patience": 2},
                       "set": {"optimizers.main.minimizes": "@losses.mae"}},
            "stop": {"when": {"use": "f_plateau", "watch": "val/rmse", "patience": 6}, "stop": True},
        },
        "plots": {"loss_curve": "f_plot", "pred_vs_true": {"use": "f_plot", "series": ["train/mse"]}},
        "record": "runs/housing",
    }


def vae():
    return {
        "data": {"use": "f_table", "file": "digits.parquet", "split": {"use": "f_kfold", "folds": 5, "fold": 1}},
        "models": {"vae": {"use": "f_ae", "latent": 8, "outputs": ["x_hat", "mu", "logvar"]}},
        "losses": {
            "mse": {"use": "f_mse", "pred": "x_hat", "target": "image"},
            "huber": {"use": "torch.nn.HuberLoss", "pred": "x_hat", "target": "image"},
            "elbo": {"use": "f_elbo", "recon": "@losses.mse", "w_rec": 1.0,
                     "w_kl": {"use": "f_warmup", "start": 0.0, "end": 1.0, "steps": 20000}},
        },
        "optimizers": {"main": {"use": "f_adam", "over": "@models.vae", "minimizes": "@losses.elbo",
                                "schedule": {"use": "f_warmup", "start": 0.0, "end": 1.0, "steps": 100}}},
        "train": {"epochs": 2},
        "rules": {"kl": {"when": {"use": "f_after_epoch", "at": 1},
                         "set": {"losses.elbo.w_rec": 0.5, "losses.elbo.recon": "@losses.huber"}}},
    }


def gan():
    return {
        "data": {"use": "f_table", "file": "faces.parquet"},
        "models": {
            "generator": {"use": "torch.nn.Linear", "in_features": 4, "out_features": 3, "inputs": ["noise"],
                          "ema": 0.999, "init": {"use": "f_init", "std": 0.02}},
            "discriminator": {"use": "torch.nn.Linear", "in_features": 3, "out_features": 1, "freeze": ["bias"]},
        },
        "losses": {"g": "f_mse", "d": "f_mse"},
        "optimizers": {
            "d": {"use": "f_adam", "lr": 2e-4,
                  "over": [{"of": "@models.discriminator", "match": "weight", "lr": 1e-4}, "@models.discriminator"],
                  "minimizes": "@losses.d"},
            "g": {"use": "f_adam", "lr": 2e-4, "over": "@models.generator", "minimizes": "@losses.g"},
        },
        "train": {"updates": {"total": 20000, "turn": 500},
                  "step": {"use": "f_alternating", "order": ["d", "g"], "repeat": {"d": 5, "g": 1},
                           "fresh_batch": True}},
        "generate": {"use": "f_sampler", "model": "@models.generator.ema", "steps": 5},
        "rules": {
            "samples": {"when": {"use": "f_every", "epochs": 5},
                        "run": {"use": "f_write_samples", "sampler": "@generate", "n": 16}},
            "thaw": {"when": {"use": "f_after_epoch", "at": 1}, "set": {"models.discriminator.freeze": False}},
        },
    }


def mdmm():
    return {
        "data": {"use": "f_table", "file": "housing.parquet"},
        "models": {"net": {"use": "torch.nn.Linear", "in_features": 3, "out_features": 1}},
        "losses": {
            "mse": "f_mse",
            "mae": "torch.nn.L1Loss",
            "total": {"use": "f_mdmm", "objective": "@losses.mse",
                      "constraints": [{"of": "@losses.mae", "epsilon": 0.05, "lmbda_init": 0.5}]},
        },
        "optimizers": {"main": {"use": "f_adam", "over": ["@models.net", {"of": "@losses.total", "lr": -0.05}],
                                "minimizes": "@losses.total"}},
        "train": {"epochs": 2, "track": ["losses.total.lmbda"]},
        "rules": {"tighten": {"when": {"use": "f_after_epoch", "at": 1},
                              "set": {"losses.total.constraints[0].epsilon": 0.02}}},
    }


def codes(tree):
    return [problem.code for problem in tree.problems]


def test_housing_tree_paths_slots_and_settings():
    config = housing()
    tree = build(config)
    assert isinstance(tree, Tree)
    assert tree.problems == []
    assert tree.order == ["data", "models.net", "losses.mse", "losses.huber", "losses.mae", "metrics.rmse",
                          "metrics.mae", "optimizers.main", "train.step", "plots.loss_curve", "plots.pred_vs_true",
                          "rules.to_huber.when", "rules.to_mae.when", "rules.stop.when"]
    assert list(tree.nodes) == tree.order
    slots = {path: node.slot for path, node in tree.nodes.items()}
    assert slots["data"] == "data"
    assert slots["models.net"] == "model"
    assert slots["losses.huber"] == "loss"
    assert slots["metrics.rmse"] == "metric"
    assert slots["optimizers.main"] == "optimizer"
    assert slots["train.step"] == "step"
    assert slots["plots.loss_curve"] == "plot"
    assert slots["rules.stop.when"] == "trigger"
    assert tree.settings == {"seed": 7, "record": "runs/housing",
                             "train": {"epochs": 3, "checkpoint": {"watch": "val/rmse"}, "report": "best",
                                       "track": ["losses.huber.delta"]}}
    assert [rule.name for rule in tree.rules] == list(config["rules"])
    assert tree.config is config
    assert list(tree.section("losses")) == ["mse", "huber", "mae"]
    assert tree.node("models.net").name == "net"
    assert tree.node("models.net").section == "models"


def test_node_use_target_keys_params_and_refs():
    tree = build(housing())
    net = tree.node("models.net")
    assert net.use == "torch.nn.Linear"
    assert net.target is nn.Linear
    assert net.keys == {}
    assert set(net.params) == {"in_features", "out_features"}
    ref = net.params["in_features"]
    assert isinstance(ref, Ref)
    assert (ref.text, ref.path, ref.attr, ref.at) == ("@data.n_features", "data", "n_features",
                                                      "models.net.in_features")
    assert net.refs == [ref]
    assert net.params["out_features"] == 1
    main = tree.node("optimizers.main")
    assert main.use == "torch.optim.Adam"
    assert main.target is torch.optim.Adam
    assert main.params == {"lr": 1e-3}
    assert set(main.keys) == {"over", "minimizes"}
    assert (main.keys["over"].path, main.keys["over"].attr) == ("models.net", None)
    assert (main.keys["minimizes"].path, main.keys["minimizes"].attr) == ("losses.mse", None)
    assert len(main.refs) == 2
    rmse = tree.node("metrics.rmse")
    assert rmse.keys == {"on": ["valid", "test"]}
    assert rmse.params == {}
    assert tree.node("losses.mse").target is mse
    assert tree.node("losses.mae").use == "torch.nn.L1Loss"
    assert tree.node("losses.huber").params == {"delta": 1.0}
    assert tree.node("data").params["features"] == {"x*": "standard_scaler"}
    assert tree.node("data").params["split"] == [0.7, 0.15, 0.15]
    plot = tree.node("plots.pred_vs_true")
    assert plot.target is fake_plot
    assert plot.params == {"series": ["train/mse"]}


def test_default_step_and_string_step():
    tree = build({"models": {"net": {"use": "torch.nn.Linear", "in_features": 2, "out_features": 1}}})
    step = tree.node("train.step")
    assert step.slot == "step"
    assert step.target is FakeStep
    assert step.params == {}
    assert tree.settings == {}
    tree = build({"train": {"epochs": 2, "step": "f_alternating"}})
    assert tree.node("train.step").target is FakeStep
    assert tree.settings == {"train": {"epochs": 2}}
    tree = build({"train": {"step": {"use": "f_alternating", "order": ["d", "g"], "repeat": {"d": 5, "g": 1},
                                     "fresh_batch": True}}})
    assert tree.problems == []
    step = tree.node("train.step")
    assert step.keys == {"order": ["d", "g"]}
    assert step.params == {"repeat": {"d": 5, "g": 1}, "fresh_batch": True}


def test_child_nodes_follow_their_parent():
    tree = build(vae())
    assert tree.problems == []
    assert tree.order[:2] == ["data", "data.split"]
    assert tree.order.index("losses.elbo.w_kl") == tree.order.index("losses.elbo") + 1
    assert tree.order.index("optimizers.main.schedule") == tree.order.index("optimizers.main") + 1
    elbo = tree.node("losses.elbo")
    child = tree.node("losses.elbo.w_kl")
    assert isinstance(child, Node)
    assert child.slot == "value"
    assert child.target is FakeWarmup
    assert child.params == {"start": 0.0, "end": 1.0, "steps": 20000}
    assert child.name == "w_kl"
    assert elbo.children == [child]
    assert elbo.params["w_kl"] is child
    assert elbo.params["w_rec"] == 1.0
    assert (elbo.params["recon"].path, elbo.params["recon"].attr) == ("losses.mse", None)
    assert [ref.at for ref in elbo.refs] == ["losses.elbo.recon"]
    split = tree.node("data.split")
    assert split.target is FakeSplit
    assert tree.node("data").params["split"] is split
    assert tree.node("data").children == [split]
    schedule = tree.node("optimizers.main.schedule")
    assert tree.node("optimizers.main").keys["schedule"] is schedule
    assert tree.node("models.vae").keys == {"outputs": ["x_hat", "mu", "logvar"]}
    assert tree.node("losses.mse").keys == {"pred": "x_hat", "target": "image"}
    assert "models" not in tree.settings


def test_reference_paths_take_the_longest_known_prefix():
    config = {
        "data": {"use": "f_table", "file": "x.parquet"},
        "models": {
            "ae": "f_ae",
            "gen": {"use": "torch.nn.Linear", "in_features": 2, "out_features": 2, "ema": 0.99},
            "probe": {"use": "f_open", "a": "@models.ae.decoder", "b": "@models.gen.ema", "c": "@data.n_classes",
                      "d": "@losses.ddpm.schedule", "e": "@generate", "f": "@optimizers.d", "g": "@models.ae",
                      "h": "@models.gen.ema.module.weight"},
        },
        "losses": {"mse": "f_mse", "ddpm": {"use": "f_elbo", "recon": "@losses.mse",
                                             "schedule": {"use": "f_warmup", "start": 0, "end": 1, "steps": 10}}},
        "optimizers": {"d": {"use": "f_adam", "over": "@models.gen"}},
        "generate": {"use": "f_sampler", "model": "@models.gen"},
    }
    tree = build(config)
    assert codes(tree) == ["unknown_key"]
    probe = tree.node("models.probe").params
    expected = {
        "a": ("models.ae", "decoder"),
        "b": ("models.gen", "ema"),
        "c": ("data", "n_classes"),
        "d": ("losses.ddpm", "schedule"),
        "e": ("generate", None),
        "f": ("optimizers.d", None),
        "g": ("models.ae", None),
        "h": ("models.gen", "ema.module.weight"),
    }
    assert {key: (ref.path, ref.attr) for key, ref in probe.items()} == expected
    assert probe["a"].name == "ae"
    assert [ref.at for ref in tree.node("models.probe").refs] == [f"models.probe.{key}" for key in expected]


def test_problems_are_recorded_with_paths_and_never_raised():
    config = {
        "data": {"use": "f_table"},
        "models": {
            "net": {"use": "torch.nn.Linear", "in_features": 2, "out_features": 1, "foo": 1},
            "short": {"use": "torch.nn.Linear", "in_features": 2},
            "clash": {"use": "f_clash", "inputs": ["x"]},
            "lost": {"use": "torch.nn.Linear", "in_features": "@models.nope", "out_features": "@nowhere.thing"},
            "unknown": "no_such_name",
            "broken": "kalfa_no_such_package.Model",
            "number": 5,
            "plain": {"in_features": 2},
            "open": {"use": "f_open", "anything": 1, "goes": "@data.n_features"},
        },
        "optimizers": {"main": {"use": "f_adam", "minimizes": "@losses.nothing"}},
        "typo_section": {"a": 1},
    }
    tree = build(config)
    found = {(problem.code, problem.path) for problem in tree.problems}
    assert ("missing_key", "data") in found
    assert ("unknown_key", "models.net") in found
    assert ("missing_key", "models.short") in found
    assert ("slot_conflict", "models.clash") in found
    assert ("reference_unknown", "models.lost") in found
    assert ("unknown_use", "models.unknown") in found
    assert ("import_failed", "models.broken") in found
    assert ("not_a_call", "models.number") in found
    assert ("not_a_call", "models.plain") in found
    assert ("missing_key", "optimizers.main") in found
    assert ("reference_unknown", "optimizers.main") in found
    assert ("unknown_section", "typo_section") in found
    assert not any(problem.path == "models.open" for problem in tree.problems)
    assert all(problem.severity == "error" for problem in tree.problems)
    assert len(tree.errors) == len(tree.problems)
    unknown = next(problem for problem in tree.problems if problem.code == "unknown_key")
    assert "'foo'" in unknown.message and "in_features" in unknown.message
    lost = [problem.message for problem in tree.problems if problem.path == "models.lost"]
    assert len(lost) == 2
    assert any("@models.nope" in message and "models.net" in message for message in lost)
    assert any("@nowhere.thing" in message for message in lost)
    assert tree.node("models.unknown").target is None
    assert tree.node("models.unknown").use == "no_such_name"
    assert tree.node("models.broken").use == "kalfa_no_such_package.Model"
    assert tree.node("models.number").target is None
    assert tree.node("models.lost").params["in_features"].path == "models.nope"
    assert tree.settings["typo_section"] == {"a": 1}
    assert tree.node("models.open").params["goes"].attr == "n_features"
    with pytest.raises(ProblemsError):
        instantiate(tree, Context(tree))


def test_bad_sections_and_reference_shapes():
    tree = build({"models": [1, 2], "rules": [1], "train": 5, "optimizers": {"o": {"use": "f_adam",
                                                                                  "over": "models.net",
                                                                                  "minimizes": "losses.mse"}}})
    assert sorted(codes(tree)) == ["bad_section", "bad_section", "bad_section", "reference_expected",
                                   "reference_expected"]
    assert tree.rules == []
    tree = build({"optimizers": {"o": {"use": "f_adam", "over": [{"match": "x"}]}}})
    assert codes(tree) == ["reference_expected"]
    tree = build(None)
    assert tree.order == ["train.step"]
    assert tree.problems == []


def test_instantiate_housing_builds_in_order(wrapped):
    tree = build(housing())
    ctx = Context(tree)
    assert instantiate(tree, ctx) is ctx
    assert isinstance(ctx.data, FakeData)
    assert ctx.data.fitted is True and ctx.data.fit_ctx is ctx
    assert ctx.data.name == "data" and ctx.data.path == "data"
    net = ctx.models["net"]
    assert isinstance(net, wrap.ModelEntry)
    assert isinstance(net.module, nn.Linear)
    assert net.module.in_features == 3
    assert net.inputs == ["x"] and net.outputs == ["net"]
    assert net.ema is None and net.freeze is False
    assert net.module.name == "net" and net.module.path == "models.net"
    assert net.path == "models.net"
    assert tree.node("models.net").obj is net.module
    assert tree.node("models.net").params["in_features"].target == 3
    assert isinstance(ctx.losses["mse"], wrap.Criterion)
    assert isinstance(ctx.losses["mse"].function, nn.MSELoss)
    assert isinstance(ctx.losses["huber"].function, nn.HuberLoss)
    assert ctx.losses["huber"].function.delta == 1.0
    assert ctx.losses["huber"].function.name == "huber"
    assert tree.node("losses.huber").obj is ctx.losses["huber"]
    assert isinstance(ctx.metrics["rmse"], wrap.MetricEntry)
    assert ctx.metrics["rmse"].on == ("valid", "test")
    assert isinstance(ctx.metrics["rmse"].metric, FakeMetric)
    main = ctx.optimizers["main"]
    assert isinstance(main, wrap.OptimizerEntry)
    assert main.target is torch.optim.Adam
    assert main.params == {"lr": 1e-3}
    assert main.over == [net.module]
    assert main.minimizes is ctx.losses["mse"]
    assert main.schedule is None
    assert isinstance(main.real, torch.optim.Adam)
    assert main.path == "optimizers.main"
    assert isinstance(ctx.step, FakeStep)
    assert ctx.step.bound is ctx
    assert ctx.step.name == "step" and ctx.step.path == "train.step"
    assert ctx.step.order is None
    assert ctx.plots == {"loss_curve": (fake_plot, {}), "pred_vs_true": (fake_plot, {"series": ["train/mse"]})}
    assert ctx.generate is None
    stop = tree.node("rules.stop.when").obj
    assert isinstance(stop, FakePlateau)
    assert stop.watch == "val/rmse" and stop.patience == 6
    assert stop.name == "when" and stop.path == "rules.stop.when"
    assert isinstance(tree.node("rules.to_huber.when").obj, FakeAfterEpoch)
    assert all(node.obj is not None for node in tree.nodes.values())


def test_instantiate_gan_models_optimizer_groups_step_and_sampler(wrapped):
    tree = build(gan())
    assert tree.problems == []
    ctx = Context(tree)
    instantiate(tree, ctx)
    generator = ctx.models["generator"]
    assert generator.inputs == ["noise"]
    assert isinstance(generator.ema, wrap.EmaCopy)
    assert generator.ema.decay == 0.999
    assert torch.all(generator.module.weight == 0.02)
    assert torch.all(generator.ema.module.weight == 0.02)
    assert isinstance(generator.init, FakeInit)
    discriminator = ctx.models["discriminator"]
    assert discriminator.asked == ["bias"]
    assert discriminator.freeze == ["bias"]
    assert discriminator.ema is None
    assert not hasattr(generator, "asked")
    d = ctx.optimizers["d"]
    assert getattr(ctx.optimizers["g"], "emas", [generator.ema]) == [generator.ema]
    assert getattr(d, "emas", []) == []
    assert len(d.over) == 2
    group, whole = d.over
    assert isinstance(group, wrap.Group)
    assert group.of is discriminator.module
    assert group.match == "weight"
    assert group.params == {"lr": 1e-4}
    assert whole is discriminator.module
    assert d.minimizes is ctx.losses["d"]
    assert ctx.optimizers["g"].over == [generator.module]
    assert isinstance(ctx.step, FakeStep)
    assert ctx.step.order == ["d", "g"]
    assert ctx.step.repeat == {"d": 5, "g": 1}
    assert ctx.step.fresh_batch is True
    assert isinstance(ctx.generate, FakeSampler)
    assert ctx.generate.model is generator.ema
    assert ctx.generate.steps == 5
    run = tree.node("rules.samples.run").obj
    assert isinstance(run, FakeWriteSamples)
    assert run.sampler is ctx.generate and run.n == 16
    assert isinstance(tree.node("rules.samples.when").obj, FakeEvery)


def test_instantiate_children_before_parents_and_binds(wrapped):
    tree = build(vae())
    ctx = Context(tree)
    instantiate(tree, ctx)
    assert isinstance(ctx.data.split, FakeSplit)
    assert ctx.data.split.fold == 1
    elbo = ctx.losses["elbo"]
    assert isinstance(elbo, FakeObjective)
    assert elbo.recon is ctx.losses["mse"]
    assert isinstance(elbo.w_kl, FakeWarmup)
    assert elbo.w_kl.steps == 20000
    assert elbo.bound is ctx
    assert elbo.name == "elbo" and elbo.path == "losses.elbo"
    assert tree.node("losses.elbo.w_kl").obj is elbo.w_kl
    assert elbo.w_kl.name == "w_kl" and elbo.w_kl.path == "losses.elbo.w_kl"
    assert isinstance(ctx.optimizers["main"].schedule, FakeWarmup)
    assert ctx.optimizers["main"].schedule.steps == 100
    assert ctx.models["vae"].outputs == ["x_hat", "mu", "logvar"]
    assert ctx.losses["mse"].pred == "x_hat" and ctx.losses["mse"].target == "image"


def test_instantiate_mdmm_lists_of_refs_and_loss_groups(wrapped):
    tree = build(mdmm())
    assert tree.problems == []
    total = tree.node("losses.total")
    assert [ref.at for ref in total.refs] == ["losses.total.objective", "losses.total.constraints[0].of"]
    ctx = Context(tree)
    instantiate(tree, ctx)
    objective = ctx.losses["total"]
    assert objective.objective is ctx.losses["mse"]
    assert objective.constraints == [{"of": ctx.losses["mae"], "epsilon": 0.05, "lmbda_init": 0.5}]
    main = ctx.optimizers["main"]
    assert main.over[0] is ctx.models["net"].module
    assert isinstance(main.over[1], wrap.Group)
    assert main.over[1].of is objective
    assert main.over[1].match is None
    assert main.over[1].params == {"lr": -0.05}
    assert len(main.real.param_groups[0]["params"]) == 3


def test_weights_are_loaded_from_another_record(wrapped, tmp_path):
    source = nn.Linear(3, 1)
    with torch.no_grad():
        source.weight.fill_(0.5)
        source.bias.fill_(-1.0)
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    torch.save({"format": 1, "turn": 2, "update": 10, "nodes": {"models.net": source.state_dict()}},
               checkpoints / "best.pt")
    config = {"models": {"net": {"use": "torch.nn.Linear", "in_features": 3, "out_features": 1,
                                 "weights": {"run": str(tmp_path), "which": "best"}},
                         "fresh": {"use": "torch.nn.Linear", "in_features": 3, "out_features": 1,
                                   "weights": {"run": str(tmp_path), "model": "net", "which": "best"}}}}
    tree = build(config)
    ctx = Context(tree)
    instantiate(tree, ctx)
    assert torch.all(ctx.models["net"].module.weight == 0.5)
    assert torch.all(ctx.models["fresh"].module.bias == -1.0)
    assert ctx.models["net"].inputs == ["x"]
    assert ctx.models["net"].weights == {"run": str(tmp_path), "which": "best"}
    bad = build({"models": {"other": {"use": "torch.nn.Linear", "in_features": 3, "out_features": 1,
                                      "weights": {"run": str(tmp_path), "which": "best"}}}})
    with pytest.raises(BuildError, match="holds no models.other"):
        instantiate(bad, Context(bad))
    missing = build({"models": {"net": {"use": "torch.nn.Linear", "in_features": 3, "out_features": 1,
                                        "weights": {"run": str(tmp_path / "nowhere")}}}})
    with pytest.raises(Exception):
        instantiate(missing, Context(missing))


def test_refs_to_unbuilt_or_hidden_targets_raise(wrapped):
    early = build({"data": {"use": "f_table", "file": "x"},
                   "models": {"net": {"use": "f_open", "loss": "@losses.mse"}},
                   "losses": {"mse": "f_mse"}})
    assert early.problems == []
    with pytest.raises(BuildError, match="not built yet"):
        instantiate(early, Context(early))
    hidden = build({"data": {"use": "f_table", "file": "x"},
                    "models": {"net": {"use": "f_open", "secret": "@data._secret"}}})
    with pytest.raises(BuildError, match="exposes"):
        instantiate(hidden, Context(hidden))
    absent = build({"data": {"use": "f_table", "file": "x"},
                    "models": {"net": {"use": "f_open", "thing": "@data.nothing_here"}}})
    with pytest.raises(BuildError, match="exposes"):
        instantiate(absent, Context(absent))
    public = build({"data": {"use": "f_table", "file": "x"},
                    "models": {"net": {"use": "f_open", "thing": "@data.public"}}})
    with pytest.raises(BuildError, match="exposes"):
        instantiate(public, Context(public))
    listed = build({"data": {"use": "f_table", "file": "x"},
                    "models": {"net": {"use": "f_open", "count": "@data.n_classes",
                                       "weights_of": "@data.class_weights"}}})
    ctx = Context(listed)
    instantiate(listed, ctx)
    assert ctx.models["net"].module.kwargs == {"count": 2, "weights_of": None}
    no_ema = build({"models": {"net": {"use": "torch.nn.Linear", "in_features": 2, "out_features": 1}},
                    "generate": {"use": "f_sampler", "model": "@models.net.ema"}})
    with pytest.raises(BuildError, match="no ema key"):
        instantiate(no_ema, Context(no_ema))
    no_attr = build({"models": {"net": {"use": "torch.nn.Linear", "in_features": 2, "out_features": 1}},
                     "generate": {"use": "f_sampler", "model": "@models.net.decoder"}})
    with pytest.raises(BuildError, match="no attribute 'decoder'"):
        instantiate(no_attr, Context(no_attr))


def test_refs_walk_attribute_chains_and_child_nodes(wrapped):
    config = {
        "models": {"ae": "f_ae"},
        "losses": {"mse": "f_mse", "ddpm": {"use": "f_elbo", "recon": "@losses.mse",
                                             "w_kl": {"use": "f_warmup", "start": 0, "end": 1, "steps": 10}}},
        "generate": {"use": "f_sampler", "model": "@models.ae.decoder", "steps": "@losses.ddpm.w_kl.steps"},
        "plots": {"p": {"use": "f_plot", "series": "@losses.ddpm.w_kl"}},
    }
    tree = build(config)
    assert tree.problems == []
    ctx = Context(tree)
    instantiate(tree, ctx)
    assert ctx.generate.model is ctx.models["ae"].module.decoder
    assert ctx.generate.steps == 10
    assert ctx.plots["p"][1] == {"series": ctx.losses["ddpm"].w_kl}
    resolved = resolve_refs(tree, ctx)
    assert resolved["generate.model"] is ctx.models["ae"].module.decoder
    assert resolve_refs(tree, ctx, ["plots.p"]) == {"plots.p.series": ctx.losses["ddpm"].w_kl}


def test_labels_skip_constructor_params_and_build_errors_name_the_node(wrapped):
    tree = build({"rules": {"r": {"when": {"use": "f_named", "name": "mine", "path": "theirs"}, "stop": True}}})
    assert tree.problems == []
    ctx = Context(tree)
    instantiate(tree, ctx)
    named = tree.node("rules.r.when").obj
    assert named.name == "mine" and named.path == "theirs"
    failing = build({"models": {"net": {"use": "torch.nn.Linear", "in_features": -1, "out_features": 1}}})
    with pytest.raises(BuildError, match="models.net: torch.nn.Linear could not be built"):
        instantiate(failing, Context(failing))


def test_kalfa_errors_from_a_constructor_carry_the_node_path_once():
    unnamed = build({"data": {"use": "table", "file": "housing.parquet", "split": [0.8, 0.2], "targets": {"x": None}}})
    with pytest.raises(BuildError, match="^data: a target cannot be named x"):
        instantiate(unnamed, Context(unnamed))
    named = build({"data": {"use": "table", "file": "housing.parquet"}})
    with pytest.raises(BuildError, match="^data.split is required"):
        instantiate(named, Context(named))


def test_mutable_paths_follow_the_contract():
    paths = mutable_paths(build(housing()))
    assert paths == sorted(paths)
    expected = {"optimizers.main.lr", "optimizers.main.weight_decay", "optimizers.main.minimizes",
                "models.net.freeze", "losses.huber.delta", "losses.huber.reduction", "losses.mse.on",
                "losses.mse.every", "metrics.rmse.on", "metrics.rmse.every", "train.epochs",
                "rules.stop.when.patience", "rules.stop.when.mode", "rules.to_huber.when.at", "train.step.order",
                "train.step.grad_clip", "train.step.repeat"}
    assert expected <= set(paths)
    unwanted = {"rules.stop.when.watch", "models.net.in_features", "models.net.inputs", "optimizers.main.over",
                "optimizers.main.params", "losses.huber.pred", "losses.huber.target", "train.updates.total",
                "plots.loss_curve.series", "data.file"}
    assert not unwanted & set(paths)
    gan_paths = set(mutable_paths(build(gan())))
    assert "train.updates.total" in gan_paths and "train.epochs" not in gan_paths
    assert {"train.step.repeat", "train.step.fresh_batch", "train.step.order", "optimizers.d.lr"} <= gan_paths
    mdmm_paths = set(mutable_paths(build(mdmm())))
    assert {"losses.total.constraints", "losses.total.constraints[0]", "losses.total.constraints[0].epsilon",
            "losses.total.constraints[0].of", "losses.total.objective"} <= mdmm_paths
    vae_paths = set(mutable_paths(build(vae())))
    assert {"losses.elbo.recon", "losses.elbo.w_rec", "losses.elbo.w_kl"} <= vae_paths
    assert "losses.elbo.w_kl.steps" not in vae_paths


class PlainData:
    def rebuild(self, ctx):
        self.rebuilt = True


class SavedData(PlainData):
    def load(self, directory):
        self.loaded = directory


class LostData(SavedData):
    def rebuild(self, ctx):
        raise KalfaError("data file 'gone.parquet' not found")

    def loaders(self):
        raise BuildError("the data is not fitted; call fit(ctx) or load(directory) first")


def test_opening_a_record_loads_what_the_data_class_saved_and_names_why_the_sets_are_missing(tmp_path):
    ctx = Context(Tree(config=None, nodes={}, order=[], settings={}, rules=[]))
    ctx.fitted_from = tmp_path
    node = SimpleNamespace(path="data", use="plain")
    plain, saved, lost = PlainData(), SavedData(), LostData()
    prepare_data(plain, ctx, node)
    prepare_data(saved, ctx, node)
    assert plain.rebuilt and saved.rebuilt and saved.loaded == str(tmp_path)
    assert ctx.rebuild_failure is None
    ctx.data = lost
    prepare_data(lost, ctx, node)
    assert lost.loaded == str(tmp_path) and ctx.rebuild_failure == "data file 'gone.parquet' not found"
    with pytest.raises(KalfaError, match="could not be rebuilt: data file 'gone.parquet' not found"):
        ctx.loaders()
