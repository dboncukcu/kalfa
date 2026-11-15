import json
from pathlib import Path

import numpy
import pyarrow
import pyarrow.parquet as parquet
import pytest
import torch
from torch import nn

from kalfa.core.aliases import ALIASES, Entry
from kalfa.core.build import Node, Ref, build, mutable_paths
from kalfa.core.plan import FORMAT, jsonable, plan_of, series_of
from kalfa.core.protocols import Objective, Step, Trigger


class FakeData:
    inputs = ("x",)
    exposes = ("n_features", "n_classes")

    def __init__(self, file, split=None, features=None, targets=None, batch=128):
        self.file = file

    def fit(self, ctx):
        pass


class FakeStep(Step):
    def __init__(self, repeat=None, fresh_batch=False):
        self.repeat = dict(repeat or {})
        self.fresh_batch = bool(fresh_batch)


class FakeObjective(Objective):
    on = ("train", "valid")
    fixed = ("recon",)

    def __init__(self, recon, w_rec=1.0, w_kl=1.0):
        super().__init__()
        self.recon = recon
        self.w_rec = w_rec
        self.w_kl = w_kl

    def forward(self, outputs, batch, ctx):
        return torch.zeros(())


def mse():
    return nn.MSELoss()


class FakeMetric:
    def reset(self):
        pass

    def update(self, pred, target):
        pass

    def compute(self):
        return 0.0


class FakePlateau(Trigger):
    fixed = ("watch",)

    def __init__(self, watch, patience, mode="min", min_delta=0.0):
        self.watch = watch
        self.patience = int(patience)

    def fires(self, ctx):
        return False


class FakeAfterEpoch(Trigger):
    def __init__(self, at):
        self.at = int(at)

    def fires(self, ctx):
        return ctx.turn >= self.at


class FakeWarmup:
    def __init__(self, start, end, steps):
        self.start = float(start)
        self.end = float(end)
        self.steps = int(steps)


class OpenModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs


def fake_plot(record, figures, series=None):
    return None


FAKES = {
    "f_table": (FakeData, "data"),
    "supervised": (FakeStep, "step"),
    "f_alternating": (FakeStep, "step"),
    "f_mse": (mse, "loss"),
    "f_elbo": (FakeObjective, "loss"),
    "f_rmse": (FakeMetric, "metric"),
    "f_plateau": (FakePlateau, "trigger"),
    "f_after_epoch": (FakeAfterEpoch, "trigger"),
    "f_warmup": (FakeWarmup, "schedule"),
    "f_open": (OpenModel, "model"),
    "f_plot": (fake_plot, "plot"),
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


def housing(file="housing.parquet"):
    return {
        "seed": 7,
        "device": "cpu",
        "data": {"use": "f_table", "file": file, "split": [0.7, 0.15, 0.15], "features": {"x*": "standard_scaler"},
                 "targets": {"price": "standard_scaler"}, "batch": 128},
        "models": {"net": {"use": "torch.nn.Linear", "in_features": "@data.n_features", "out_features": 1},
                   "open": {"use": "f_open", "anything": (1, 2)}},
        "losses": {"mse": "f_mse", "huber": {"use": "torch.nn.HuberLoss", "delta": 1.0, "on": ["train"]},
                   "elbo": {"use": "f_elbo", "recon": "@losses.mse",
                            "w_kl": {"use": "f_warmup", "start": 0.0, "end": 1.0, "steps": 20000}}},
        "metrics": {"rmse": {"use": "f_rmse", "on": ["valid", "test"]}, "mae": "f_rmse"},
        "optimizers": {"main": {"use": "f_adam", "lr": 1e-3, "over": "@models.net", "minimizes": "@losses.mse"},
                       "aux": {"use": "f_adam", "over": [{"of": "@models.net", "match": "weight", "lr": 1e-4}],
                               "schedule": {"use": "f_warmup", "start": 0.0, "end": 1.0, "steps": 100}}},
        "train": {"epochs": 3, "checkpoint": {"watch": "val/rmse"}, "report": "best",
                  "track": ["losses.huber.delta", "losses.elbo.w_rec"], "step": "f_alternating"},
        "rules": {
            "to_huber": {"when": {"use": "f_after_epoch", "at": 2},
                         "set": {"optimizers.main.minimizes": "@losses.huber", "losses.elbo.w_rec": {"times": 0.5}}},
            "stop": {"when": {"use": "f_plateau", "watch": "val/rmse", "patience": 6}, "stop": True,
                     "after": "to_huber", "once": False},
            "always": {"set": {"losses.huber.delta": 2.0}},
        },
        "plots": {"loss_curve": "f_plot"},
        "record": "runs/housing",
    }


def test_plan_shape_and_node_entries():
    tree = build(housing())
    assert tree.problems == []
    plan = plan_of(tree, version="3.0.0.test")
    assert list(plan) == ["format", "kalfa", "seed", "device", "record", "nodes", "data", "train", "rules",
                          "series", "mutable"]
    assert plan["format"] == FORMAT == 1
    assert plan["kalfa"] == "3.0.0.test"
    assert plan["seed"] == 7
    assert plan["device"] == "cpu"
    assert plan["record"] == "runs/housing"
    assert list(plan["nodes"]) == tree.order
    net = plan["nodes"]["models.net"]
    assert net["slot"] == "model"
    assert net["use"] == "torch.nn.Linear"
    assert net["params"] == {"in_features": {"ref": "data", "attr": "n_features"}, "out_features": 1}
    assert net["keys"] == {}
    assert net["signature"] == "checked"
    assert {"inputs", "outputs", "ema", "weights", "init", "in_features", "out_features", "bias"} <= set(net["fixed"])
    assert "freeze" not in net["fixed"]
    open_model = plan["nodes"]["models.open"]
    assert open_model["signature"] == "open"
    assert open_model["params"] == {"anything": [1, 2]}
    elbo = plan["nodes"]["losses.elbo"]
    assert elbo["params"] == {"recon": {"ref": "losses.mse"}, "w_kl": {"node": "losses.elbo.w_kl"}}
    assert elbo["fixed"] == ["pred", "target", "recon"]
    child = plan["nodes"]["losses.elbo.w_kl"]
    assert child == {"slot": "value", "use": f"{FakeWarmup.__module__}.FakeWarmup",
                     "params": {"start": 0.0, "end": 1.0, "steps": 20000}, "keys": {}, "fixed": [],
                     "signature": "checked"}
    huber = plan["nodes"]["losses.huber"]
    assert huber["keys"] == {"on": ["train"]}
    assert huber["params"] == {"delta": 1.0}
    assert huber["fixed"] == ["pred", "target"]
    main = plan["nodes"]["optimizers.main"]
    assert main["use"] == "torch.optim.Adam"
    assert main["keys"] == {"over": {"ref": "models.net"}, "minimizes": {"ref": "losses.mse"}}
    assert main["fixed"] == ["over", "schedule"]
    aux = plan["nodes"]["optimizers.aux"]
    assert aux["keys"] == {"over": [{"of": {"ref": "models.net"}, "match": "weight", "lr": 1e-4}],
                           "schedule": {"node": "optimizers.aux.schedule"}}
    step = plan["nodes"]["train.step"]
    assert step["slot"] == "step" and step["use"].endswith("FakeStep") and step["fixed"] == []
    assert plan["nodes"]["rules.stop.when"]["fixed"] == ["watch"]
    assert plan["nodes"]["plots.loss_curve"]["signature"] == "checked"
    assert plan["train"] == {"epochs": 3, "checkpoint": {"watch": "val/rmse"}, "report": "best",
                             "track": ["losses.huber.delta", "losses.elbo.w_rec"]}
    assert plan["mutable"] == mutable_paths(tree)
    assert "optimizers.main.lr" in plan["mutable"]


def test_plan_data_section_without_a_readable_file():
    plan = plan_of(build(housing()))
    assert plan["data"] == {"inputs": ["x"], "exposes": ["n_features", "n_classes"]}
    assert plan["kalfa"].startswith("3.")
    empty = plan_of(build({"train": {"epochs": 1}}))
    assert empty["data"] == {"inputs": [], "exposes": []}
    assert empty["nodes"].keys() == {"train.step"}
    assert empty["rules"] == []
    assert empty["series"] == []
    assert empty["seed"] is None


def test_plan_reads_parquet_and_csv_headers(tmp_path):
    table = pyarrow.table({"x1": [1.0, 2.0], "x2": [3, 4], "price": [5.0, 6.0]})
    parquet.write_table(table, tmp_path / "housing.parquet")
    plan = plan_of(build(housing(str(tmp_path / "housing.parquet"))))
    assert plan["data"]["fields"] == {"x1": "double", "x2": "int64", "price": "double"}
    (tmp_path / "rows.csv").write_text("a,b\n1,2.5\n3,4.5\n", encoding="utf-8")
    plan = plan_of(build(housing(str(tmp_path / "rows.csv"))))
    assert plan["data"]["fields"] == {"a": "int64", "b": "double"}
    (tmp_path / "rows.tsv").write_text("c\td\n1\tx\n", encoding="utf-8")
    plan = plan_of(build(housing(str(tmp_path / "rows.tsv"))))
    assert plan["data"]["fields"] == {"c": "int64", "d": "string"}
    plan = plan_of(build(housing(str(tmp_path / "missing.parquet"))))
    assert "fields" not in plan["data"]
    (tmp_path / "notes.txt").write_text("hello\n", encoding="utf-8")
    plan = plan_of(build(housing(str(tmp_path / "notes.txt"))))
    assert "fields" not in plan["data"]


def test_plan_resolves_the_file_against_config_directories(tmp_path):
    class Config:
        def __init__(self, data, directories):
            self.data = data
            self.directories = directories

    parquet.write_table(pyarrow.table({"only": [1]}), tmp_path / "rel.parquet")
    plan = plan_of(build(Config(housing("rel.parquet"), [str(tmp_path)])))
    assert plan["data"]["fields"] == {"only": "int64"}


def test_rules_in_plan():
    plan = plan_of(build(housing()))
    assert plan["rules"] == [
        {"id": "to_huber", "when": {"use": "f_after_epoch", "at": 2}, "after": None, "once": True,
         "patch": [{"path": "optimizers.main.minimizes", "value": {"ref": "losses.huber"}},
                   {"path": "losses.elbo.w_rec", "value": {"times": 0.5}}], "stop": False, "run": None},
        {"id": "stop", "when": {"use": "f_plateau", "watch": "val/rmse", "patience": 6}, "after": "to_huber",
         "once": False, "patch": [], "stop": True, "run": None},
        {"id": "always", "when": None, "after": None, "once": True,
         "patch": [{"path": "losses.huber.delta", "value": 2.0}], "stop": False, "run": None},
    ]
    plan = plan_of(build({"rules": {"s": {"when": "f_after_epoch", "run": "f_plot"}}}))
    assert plan["rules"][0]["when"] == {"use": "f_after_epoch"}
    assert plan["rules"][0]["run"] == {"use": "f_plot"}
    config = housing()
    config["generate"] = {"use": "f_warmup", "start": 0.0, "end": 1.0, "steps": 5}
    config["rules"] = {"samples": {"when": {"use": "f_after_epoch", "at": 1},
                                   "run": {"use": "f_plot", "series": "@generate", "n": 16}}}
    plan = plan_of(build(config))
    assert plan["rules"][0]["run"] == {"use": "f_plot", "series": {"ref": "generate"}, "n": 16}


def test_series_follow_losses_metrics_optimizers_and_track():
    tree = build(housing())
    series = series_of(tree)
    assert series == plan_of(tree)["series"]
    assert series == [
        "train/mse", "val/mse", "test/mse",
        "train/huber",
        "train/elbo", "val/elbo",
        "val/rmse", "test/rmse",
        "train/mae", "val/mae", "test/mae",
        "lr/main", "minimizes/main", "lr/aux", "minimizes/aux",
        "track/losses.huber.delta", "track/losses.elbo.w_rec",
    ]
    config = housing()
    config["train"]["track"] = "losses.huber.delta"
    config["metrics"]["rmse"]["on"] = "valid"
    assert series_of(build(config))[6:8] == ["val/rmse", "train/mae"]
    assert series_of(build(config))[-1] == "track/losses.huber.delta"


def test_jsonable_converts_refs_nodes_tuples_and_unknowns():
    ref = Ref("@models.ae.decoder", "models.ae", "decoder", "generate.model")
    plain = Ref("@generate", "generate", None, "rules.s.run.sampler")
    node = Node("losses.elbo.w_kl", "value", "x.Warmup", None, {})
    value = {"a": ref, "b": plain, "c": node, "d": (1, 2), "e": {"f": [ref, None, True, 1.5]}, "g": Path("p/q"),
             "h": numpy.int64(3), "i": numpy.float32(0.5), "j": {1: "one"}, "k": frozenset([1])}
    out = jsonable(value)
    assert out == {"a": {"ref": "models.ae", "attr": "decoder"}, "b": {"ref": "generate"},
                   "c": {"node": "losses.elbo.w_kl"}, "d": [1, 2],
                   "e": {"f": [{"ref": "models.ae", "attr": "decoder"}, None, True, 1.5]}, "g": "p/q", "h": 3,
                   "i": 0.5, "j": {"1": "one"}, "k": [1]}
    assert isinstance(out["h"], int) and isinstance(out["i"], float)
    assert jsonable(torch.tensor(2.0)) == 2.0
    assert jsonable(torch.zeros(2)) == [0.0, 0.0]
    assert jsonable("@text") == "@text"
    json.dumps(out)
