import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import kalfa
from helpers import housing_frame
from kalfa.core import check as checking, graph, loop, run as running


class Branchy(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = nn.Linear(4, 8)
        self.right = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
        self.head = nn.Linear(16, 2)

    def forward(self, x):
        joined = torch.cat([self.left(x), self.right(x)], dim=-1)
        return self.head(joined) + 1.0


class TwoHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Linear(3, 5)
        self.mu = nn.Linear(5, 2)
        self.logvar = nn.Linear(5, 2)

    def forward(self, x):
        hidden = torch.relu(self.body(x)).view(x.shape[0], -1)
        return {"mu": self.mu(hidden), "logvar": self.logvar(hidden)}


class Twice(nn.Module):
    def __init__(self):
        super().__init__()
        self.step = nn.Linear(3, 3)

    def forward(self, x):
        return self.step(self.step(x))


class Broken(nn.Module):
    def forward(self, x):
        raise RuntimeError("no forward")


class Denoiser(nn.Module):
    def __init__(self, width=4, steps=10):
        super().__init__()
        self.time = nn.Embedding(steps, width)
        self.body = nn.Linear(width, width)

    def forward(self, x, t):
        return self.body(x + self.time(t))


class Watcher:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.seen = []

    def expect(self, total):
        return None

    def say(self, text):
        return None

    def turn(self, line):
        self.seen.append((self.directory / "model_summary.json").is_file())


class GraphWatcher(Watcher):
    def turn(self, line):
        summary = json.loads((self.directory / "model_summary.json").read_text(encoding="utf-8"))
        self.seen.append(summary["unet"]["graph"] is not None)


def of_kind(found, kind):
    return [node for node in found["nodes"] if node["kind"] == kind]


def links(found):
    return {(edge["from"], edge["to"]) for edge in found["edges"]}


def housing_config(home, epochs=1):
    return {
        "seed": 3, "device": "cpu",
        "data": {"use": "table", "file": str(home / "housing.parquet"), "split": [0.7, 0.15, 0.15],
                 "features": {"x*": "standard_scaler"}, "targets": {"price": "standard_scaler"}, "batch": 32},
        "models": {"net": {"use": "mlp", "hidden": [8], "out": 1}},
        "losses": {"mse": "mse"}, "metrics": {"rmse": "rmse"},
        "optimizers": {"main": {"use": "adam", "lr": 0.01, "over": "@models.net", "minimizes": "@losses.mse"}},
        "train": {"epochs": epochs},
    }


@pytest.fixture(scope="module")
def home(tmp_path_factory):
    root = tmp_path_factory.mktemp("phase6_graph")
    housing_frame(120, 0).to_parquet(root / "housing.parquet", index=False)
    return root


def test_a_branching_model_gives_modules_ops_groups_and_edges():
    found = graph.trace(Branchy(), (torch.randn(5, 4),), ["x"], ["y"])
    assert found["method"] == "eager" and found["truncated"] is False
    assert len({node["id"] for node in found["nodes"]}) == len(found["nodes"])
    modules = {node["path"]: node for node in of_kind(found, "module")}
    assert set(modules) == {"left", "right.0", "right.1", "head"}
    assert modules["left"]["label"] == "Linear" and modules["left"]["parameters"] == 40
    assert modules["left"]["class"].endswith("Linear") and modules["left"]["detail"] == "left"
    assert modules["right.1"]["label"] == "ReLU" and modules["right.1"]["parameters"] == 0
    assert modules["head"]["parameters"] == 34
    ops = {node["label"]: node for node in of_kind(found, "op")}
    assert sorted(ops) == ["add", "cat"]
    [source] = of_kind(found, "input")
    [sink] = of_kind(found, "output")
    assert source["label"] == "x" and source["shape"] == [4]
    assert sink["label"] == "y" and sink["shape"] == [2]
    assert ops["cat"]["shape"] == [16] and modules["right.0"]["shape"] == [8]
    expected = {(source["id"], modules["left"]["id"]), (source["id"], modules["right.0"]["id"]),
                (modules["right.0"]["id"], modules["right.1"]["id"]), (modules["left"]["id"], ops["cat"]["id"]),
                (modules["right.1"]["id"], ops["cat"]["id"]), (ops["cat"]["id"], modules["head"]["id"]),
                (modules["head"]["id"], ops["add"]["id"]), (ops["add"]["id"], sink["id"])}
    assert links(found) == expected
    shapes = {(edge["from"], edge["to"]): edge["shape"] for edge in found["edges"]}
    assert shapes[(ops["cat"]["id"], modules["head"]["id"])] == [16]
    [group] = found["groups"]
    assert group["path"] == "right" and group["label"] == "right" and group["detail"] == "Sequential"
    assert group["class"].endswith("Sequential") and group["parameters"] == 40 and group["parent"] is None
    assert modules["right.0"]["group"] == group["id"] == modules["right.1"]["group"]
    assert modules["left"]["group"] is None and ops["cat"]["group"] is None
    json.dumps(found, allow_nan=False)


def test_dict_outputs_functional_activations_and_views_that_pass_through():
    found = graph.trace(TwoHeads(), (torch.randn(2, 3),), ["x"], ["mu", "logvar"])
    modules = {node["path"]: node for node in of_kind(found, "module")}
    ops = of_kind(found, "op")
    assert [node["label"] for node in ops] == ["relu"]
    outputs = {node["label"]: node for node in of_kind(found, "output")}
    assert set(outputs) == {"mu", "logvar"} and outputs["mu"]["shape"] == [2]
    relu = ops[0]["id"]
    assert links(found) >= {(modules["body"]["id"], relu), (relu, modules["mu"]["id"]),
                            (relu, modules["logvar"]["id"]), (modules["mu"]["id"], outputs["mu"]["id"]),
                            (modules["logvar"]["id"], outputs["logvar"]["id"])}
    assert not [node for node in found["nodes"] if node["label"] in ("view", "reshape")]


def test_a_module_called_twice_gives_two_nodes():
    found = graph.trace(Twice(), (torch.randn(2, 3),), ["x"], ["y"])
    steps = of_kind(found, "module")
    assert [node["path"] for node in steps] == ["step", "step"] and steps[0]["id"] != steps[1]["id"]
    assert (steps[0]["id"], steps[1]["id"]) in links(found)


def test_the_limit_truncates_and_keeps_edges_between_kept_nodes():
    found = graph.trace(Branchy(), (torch.randn(5, 4),), ["x"], ["y"], limit=2)
    assert found["truncated"] is True
    assert len([node for node in found["nodes"] if node["kind"] in ("module", "op")]) <= 2
    kept = {node["id"] for node in found["nodes"]}
    assert all(edge["from"] in kept and edge["to"] in kept for edge in found["edges"])


def test_a_failing_forward_gives_none():
    assert graph.trace(Broken(), (torch.randn(2, 3),), ["x"], ["y"]) is None


def test_the_model_summary_carries_the_graph_and_is_written_before_the_first_turn(home):
    target = home / "runs" / "net"
    watcher = Watcher(target)
    result = kalfa.run(housing_config(home, epochs=2), record=str(target), monitor=watcher)
    assert watcher.seen and watcher.seen[0] is True
    summary = json.loads((Path(result.record) / "model_summary.json").read_text(encoding="utf-8"))
    found = summary["net"]["graph"]
    assert [node["label"] for node in found["nodes"] if node["kind"] == "input"] == ["x"]
    assert [node["label"] for node in found["nodes"] if node["kind"] == "output"] == ["net"]
    assert of_kind(found, "module")
    assert summary["net"]["tree"] and "net" in summary["net"]["outputs"]


def test_the_summary_keeps_every_random_stream(home):
    checked = checking.check(mappings=[housing_config(home)])
    assert checked.ok, checked.text()
    ctx = running.prepare(checked, record=False)
    ctx.loaders()
    torch_state = torch.random.get_rng_state().clone()
    streams = ctx.rng.state_dict()["streams"]
    assert streams
    loop.model_summary(ctx)
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    after = ctx.rng.state_dict()["streams"]
    for key, state in streams.items():
        assert torch.equal(after[key], state), key


def denoiser_config(home, epochs=2):
    return {
        "seed": 3, "device": "cpu",
        "data": {"use": "table", "file": str(home / "housing.parquet"), "split": [0.7, 0.15, 0.15],
                 "features": {"x*": "standard_scaler"}, "targets": {"price": "standard_scaler"}, "batch": 32},
        "models": {"unet": {"use": "test_phase6_graph.Denoiser", "width": 4, "steps": 10}},
        "losses": {"ddpm": {"use": "ddpm", "model": "@models.unet", "x": "x",
                            "schedule": {"use": "cosine_beta", "steps": 10}}},
        "optimizers": {"main": {"use": "adam", "lr": 0.01, "over": "@models.unet", "minimizes": "@losses.ddpm"}},
        "train": {"epochs": epochs, "predicts": False},
    }


def test_keyword_inputs_are_traced_and_named():
    found = graph.trace(Denoiser(), (torch.randn(3, 4),), ["x"], ["y"], keywords={"t": torch.tensor([1, 2, 3])})
    inputs = {node["label"]: node for node in of_kind(found, "input")}
    assert set(inputs) == {"x", "t"} and inputs["t"]["shape"] == [] and inputs["t"]["detail"] == "int64"
    assert graph.argument_names(Denoiser(), 3) == ["x", "t", "input_2"]


def test_a_model_a_loss_calls_with_its_own_inputs_is_drawn_from_its_first_training_call(home):
    target = home / "runs" / "denoiser"
    watcher = GraphWatcher(target)
    kalfa.run(denoiser_config(home), record=str(target), monitor=watcher)
    assert watcher.seen == [False, True]
    summary = json.loads((target / "model_summary.json").read_text(encoding="utf-8"))["unet"]
    assert summary["graph_from"] == "first training call" and "graph_note" not in summary
    found = summary["graph"]
    inputs = {node["label"]: node for node in of_kind(found, "input")}
    assert set(inputs) == {"x", "t"} and inputs["x"]["shape"] == [4] and inputs["t"]["shape"] == []
    assert {node["path"] for node in of_kind(found, "module")} == {"time", "body"}
    assert [node["label"] for node in of_kind(found, "output")] == ["unet"] and "unet" in summary["outputs"]


def test_drawing_from_the_first_training_call_leaves_training_unchanged(home, monkeypatch):
    first = kalfa.run(denoiser_config(home, epochs=3), record=str(home / "runs" / "drawn"))
    monkeypatch.setattr(loop.FirstCalls, "helps", lambda self, summary: False)
    monkeypatch.setattr(loop.FirstCalls, "keeper", lambda self, name: lambda module, args, kwargs: None)
    second = kalfa.run(denoiser_config(home, epochs=3), record=str(home / "runs" / "plain"))
    drop = ("seconds", "time", "at")
    assert [{key: value for key, value in line.items() if key not in drop} for line in first.history] == [
        {key: value for key, value in line.items() if key not in drop} for line in second.history]


def test_a_captured_half_precision_input_is_cast_to_the_model_dtype():
    entry = SimpleNamespace(module=nn.Linear(4, 2), outputs=["y"], name_outputs=lambda result: {"y": result})
    call = ((torch.randn(3, 4).half(),), {})
    found = loop.called_graph("net", entry, call)
    assert found is not None and [node["shape"] for node in of_kind(found, "output")] == [[2]]
    assert loop.called_outputs(entry, call) == {"y": [3, 2]}
