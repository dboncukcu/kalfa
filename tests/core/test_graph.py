import json
import weakref
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional
from torch.nn.parameter import is_lazy

import kalfa
from helpers import housing_frame
from kalfa.core import graph, loop
from kalfa.core.context import Context
from kalfa.core.rng import Rng
from kalfa.core.wrap import ModelEntry
from kalfa.std.init import normal_init


class Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Linear(4, 4)
        self.act = nn.Tanh()

    def forward(self, x):
        return self.act(self.scale(x)) * 2


class Outer(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(Inner(), nn.Linear(4, 4))
        self.last = nn.Linear(4, 3)

    def forward(self, x):
        return self.last(1 - self.stack(x))


class Pair(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Linear(3, 4)
        self.a = nn.Linear(4, 2)
        self.b = nn.Linear(4, 5)

    def forward(self, x):
        hidden = self.body(x)
        hidden.mul_(0.5)
        return self.a(hidden), self.b(hidden), torch.sigmoid(hidden)


class Attention(nn.Module):
    def __init__(self, dim=8, heads=2):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        count, steps, dim = x.shape
        q, k, v = self.qkv(x).split(dim, dim=-1)
        q, k, v = (part.view(count, steps, self.heads, dim // self.heads).transpose(1, 2) for part in (q, k, v))
        mixed = functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(mixed.transpose(1, 2).reshape(count, steps, dim))


class Block(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = Attention(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 16), nn.GELU(), nn.Linear(16, dim))

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        scores = torch.softmax(x @ x.transpose(-2, -1), dim=-1)
        x = x + torch.matmul(scores, x)
        return x + self.mlp(x)


class Words(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(20, 6)
        self.place = nn.Embedding(7, 6)
        self.head = nn.Linear(6, 20)

    def forward(self, tokens):
        steps = tokens.shape[1]
        hidden = self.embed(tokens) + self.place(torch.arange(steps, device=tokens.device))
        return self.head(functional.gelu(hidden))


class Shared(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        self.head = nn.Linear(4, 1)

    def forward(self, x):
        return self.head(self.block(self.block(x)))


class Deep(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(*[nn.Sequential(nn.Linear(4, 4), nn.ReLU()) for _ in range(30)])

    def forward(self, x):
        return self.layers(x) + x


class TwoInputs(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(2, 1)

    def forward(self, x, t):
        return self.net(x) + t


class Forgetful(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
        self.freed = None

    def forward(self, x):
        kept = []
        for layer in self.layers:
            x = torch.relu(layer(x))
            kept.append(weakref.ref(x))
        self.freed = [ref() is None for ref in kept[:-1]]
        return x


class Odd(nn.Module):
    def parameters(self, recurse=True):
        raise RuntimeError("no parameters to count")

    def forward(self, x):
        return x


class Rows(torch.utils.data.Dataset):
    def __init__(self, count):
        self.x = torch.randn(count, 2, generator=torch.Generator().manual_seed(0))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return {"x": self.x[index]}


class Streamed(torch.utils.data.IterableDataset):
    def __init__(self, rng):
        self.rng = rng

    def __iter__(self):
        order = torch.randperm(8, generator=self.rng.stream("lazy"))
        for index in order.tolist():
            yield {"x": torch.full((2,), float(index))}


class Data:
    inputs = ("x",)

    def __init__(self, rng, streamed=False):
        self.rng = rng
        self.streamed = streamed
        self.made = None

    def loaders(self):
        if self.made is None:
            self.made = {"train": torch.utils.data.DataLoader(Rows(12), batch_size=4, shuffle=True,
                                                              generator=self.rng.stream("loader"))}
            if self.streamed:
                self.made["valid"] = torch.utils.data.DataLoader(Streamed(self.rng), batch_size=4)
        return dict(self.made)


def of_kind(found, kind):
    return [node for node in found["nodes"] if node["kind"] == kind]


def by_path(found):
    return {node["path"]: node for node in of_kind(found, "module")}


def links(found):
    return {(edge["from"], edge["to"]) for edge in found["edges"]}


def well_formed(found):
    ids = [node["id"] for node in found["nodes"]]
    groups = [group["id"] for group in found["groups"]]
    assert len(set(ids)) == len(ids) and len(set(groups)) == len(groups)
    assert all(edge["from"] in ids and edge["to"] in ids for edge in found["edges"])
    assert all(node["group"] is None or node["group"] in groups for node in found["nodes"])
    for position, group in enumerate(found["groups"]):
        assert group["parent"] is None or group["parent"] in groups[:position]
    json.dumps(found, allow_nan=False)
    return found


def hooks_left(module):
    return sum(len(part._forward_hooks) + len(part._forward_pre_hooks) for part in module.modules())


def summary_context(module, streamed=False, inputs=("x",)):
    ctx = Context(SimpleNamespace(config={}, settings={"train": {"epochs": 1}}))
    ctx.rng = Rng(5)
    ctx.data = Data(ctx.rng, streamed)
    ctx.models = {"net": ModelEntry("net", module, list(inputs), ["net"])}
    return ctx


def test_nested_custom_blocks_become_nested_groups():
    found = well_formed(graph.trace(Outer(), (torch.randn(3, 4),), ["x"], ["y"]))
    groups = {group["path"]: group for group in found["groups"]}
    assert set(groups) == {"stack", "stack.0"}
    assert groups["stack"]["parent"] is None and groups["stack.0"]["parent"] == groups["stack"]["id"]
    assert groups["stack"]["id"] != groups["stack.0"]["id"]
    assert groups["stack.0"]["label"] == "stack.0" and groups["stack.0"]["detail"] == "Inner"
    assert groups["stack"]["parameters"] == 40 and groups["stack.0"]["parameters"] == 20
    modules = by_path(found)
    assert set(modules) == {"stack.0.scale", "stack.0.act", "stack.1", "last"}
    assert modules["stack.0.scale"]["group"] == groups["stack.0"]["id"]
    assert modules["stack.1"]["group"] == groups["stack"]["id"] and modules["last"]["group"] is None
    ops = {node["label"]: node for node in of_kind(found, "op")}
    assert set(ops) == {"mul", "sub"}
    assert ops["mul"]["group"] == groups["stack.0"]["id"] and ops["sub"]["group"] is None
    assert links(found) >= {(modules["stack.0.act"]["id"], ops["mul"]["id"]),
                            (ops["mul"]["id"], modules["stack.1"]["id"]),
                            (modules["stack.1"]["id"], ops["sub"]["id"]), (ops["sub"]["id"], modules["last"]["id"])}


def test_tuple_outputs_are_named_in_order_and_in_place_ops_keep_the_chain():
    found = well_formed(graph.trace(Pair(), (torch.randn(6, 3),), ["x"], ["first", "second"]))
    outputs = {node["label"]: node for node in of_kind(found, "output")}
    assert set(outputs) == {"first", "second", "output_2"}
    assert outputs["first"]["shape"] == [2] and outputs["second"]["shape"] == [5] and outputs["output_2"]["shape"] == [4]
    modules = by_path(found)
    ops = {node["label"]: node for node in of_kind(found, "op")}
    assert set(ops) == {"mul", "sigmoid"}
    assert links(found) >= {(modules["body"]["id"], ops["mul"]["id"]), (ops["mul"]["id"], modules["a"]["id"]),
                            (ops["mul"]["id"], modules["b"]["id"]), (ops["mul"]["id"], ops["sigmoid"]["id"]),
                            (modules["a"]["id"], outputs["first"]["id"]),
                            (ops["sigmoid"]["id"], outputs["output_2"]["id"])}


def test_a_transformer_block_shows_attention_and_matmul_without_views():
    found = well_formed(graph.trace(Block(), (torch.randn(3, 5, 8),), ["x"], ["y"]))
    labels = [node["label"] for node in of_kind(found, "op")]
    assert labels.count("scaled_dot_product_attention") == 1 and labels.count("matmul") == 2
    assert "softmax" in labels and "add" in labels
    assert not {"view", "transpose", "split", "reshape", "getitem"} & set(labels)
    modules = by_path(found)
    [attention] = [node for node in of_kind(found, "op") if node["label"] == "scaled_dot_product_attention"]
    groups = {group["path"]: group for group in found["groups"]}
    assert attention["group"] == groups["attn"]["id"] and attention["shape"] == [2, 5, 4]
    assert attention["detail"] == "nn.functional.scaled_dot_product_attention"
    assert (modules["attn.qkv"]["id"], attention["id"]) in links(found)
    assert (attention["id"], modules["attn.proj"]["id"]) in links(found)
    shapes = {(edge["from"], edge["to"]): edge["shape"] for edge in found["edges"]}
    assert shapes[(attention["id"], modules["attn.proj"]["id"])] == [5, 8]
    [source] = of_kind(found, "input")
    assert source["shape"] == [5, 8] and source["detail"] == "float32"
    assert of_kind(found, "output")[0]["shape"] == [5, 8]


def test_an_embedding_model_takes_integer_inputs():
    tokens = torch.randint(0, 20, (4, 7))
    found = well_formed(graph.trace(Words(), (tokens,), ["tokens"], ["logits"]))
    [source] = of_kind(found, "input")
    assert source["label"] == "tokens" and source["detail"] == "int64" and source["shape"] == [7]
    modules = by_path(found)
    assert modules["embed"]["shape"] == [7, 6] and modules["embed"]["parameters"] == 120
    assert (source["id"], modules["embed"]["id"]) in links(found)
    assert not [edge for edge in found["edges"] if edge["to"] == modules["place"]["id"]]
    ops = {node["label"]: node for node in of_kind(found, "op")}
    assert set(ops) == {"add", "gelu"}
    assert (modules["place"]["id"], ops["add"]["id"]) in links(found)
    assert of_kind(found, "output")[0]["shape"] == [7, 20]


def test_a_block_used_twice_gives_two_groups_and_two_node_sets():
    found = well_formed(graph.trace(Shared(), (torch.randn(3, 4),), ["x"], ["y"]))
    groups = found["groups"]
    assert [group["path"] for group in groups] == ["block", "block"] and groups[0]["id"] != groups[1]["id"]
    linear = [node for node in of_kind(found, "module") if node["path"] == "block.0"]
    assert len(linear) == 2 and [node["group"] for node in linear] == [groups[0]["id"], groups[1]["id"]]
    relu = [node for node in of_kind(found, "module") if node["path"] == "block.1"]
    assert (relu[0]["id"], linear[1]["id"]) in links(found)


def test_the_limit_keeps_the_first_nodes_and_their_groups():
    found = well_formed(graph.trace(Deep(), (torch.randn(3, 4),), ["x"], ["y"], limit=10))
    assert found["truncated"] is True
    kept = [node for node in found["nodes"] if node["kind"] in ("module", "op")]
    assert len(kept) == 10 and [node["path"] for node in kept[:2]] == ["layers.0.0", "layers.0.1"]
    names = {node["id"] for node in found["nodes"]}
    assert all(edge["from"] in names and edge["to"] in names for edge in found["edges"])
    used = {node["group"] for node in found["nodes"]}
    assert {group["id"] for group in found["groups"]} <= used | {group["parent"] for group in found["groups"]}
    assert len(found["groups"]) == 6
    assert [node["label"] for node in of_kind(found, "output")] == ["y"]
    whole = graph.trace(Deep(), (torch.randn(3, 4),), ["x"], ["y"])
    assert whole["truncated"] is False and len(of_kind(whole, "module")) == 60


def test_hooks_are_removed_after_success_and_failure():
    model = Outer()
    assert graph.trace(model, (torch.randn(2, 4),), ["x"], ["y"]) is not None
    assert hooks_left(model) == 0
    assert graph.trace(model, (torch.randn(2, 5),), ["x"], ["y"]) is None
    assert hooks_left(model) == 0
    assert torch.randn(2, 4).add(1).shape == (2, 4)


def kept_streams(streamed):
    ctx = summary_context(nn.Linear(2, 1), streamed=streamed)
    ctx.loaders()
    ctx.rng.stream("noise")
    before = {key: state.clone() for key, state in ctx.rng.state_dict()["streams"].items()}
    torch_state = torch.random.get_rng_state().clone()
    summary = loop.model_summary(ctx)
    after = ctx.rng.state_dict()["streams"]
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    for key, state in before.items():
        assert torch.equal(after[key], state), key
    return summary, before, after


def test_the_summary_reads_a_train_only_loader_without_moving_its_stream():
    summary, before, after = kept_streams(False)
    assert set(after) == set(before) == {"loader@cpu", "noise@cpu"}
    found = summary["net"]["graph"]
    assert [node["label"] for node in found["nodes"]] == ["x", "Linear", "net"]
    assert summary["net"]["outputs"] == {"net": [4, 1]} and "graph_note" not in summary["net"]


def test_a_stream_made_while_reading_the_first_batch_starts_fresh_afterwards():
    summary, before, after = kept_streams(True)
    assert set(after) - set(before) == {"lazy@cpu"}
    assert torch.equal(after["lazy@cpu"], Rng(5).stream("lazy").get_state())
    assert summary["net"]["graph"]["nodes"][0]["shape"] == [2]


def test_a_model_that_needs_other_inputs_has_no_graph_and_a_note():
    ctx = summary_context(TwoInputs())
    summary = loop.model_summary(ctx)
    assert summary["net"]["graph"] is None and summary["net"]["outputs"] == {}
    assert "could not run on the first batch" in summary["net"]["graph_note"]
    assert summary["net"]["tree"][0]["name"] == "net"


def test_the_trace_lets_go_of_tensors_the_model_no_longer_holds():
    model = Forgetful()
    found = well_formed(graph.trace(model, (torch.randn(3, 4),), ["x"], ["y"]))
    assert model.freed == [True, True]
    assert [node["label"] for node in found["nodes"]] == ["x", "Linear", "relu", "Linear", "relu", "Linear", "relu",
                                                          "y"]
    ids = [node["id"] for node in found["nodes"]]
    assert links(found) == set(zip(ids, ids[1:]))


def test_the_summary_runs_a_lazy_model_on_a_copy_and_leaves_it_lazy():
    module = nn.Sequential(nn.LazyLinear(3), nn.ReLU(), nn.LazyLinear(1))
    normal_init(std=0.5)(module)
    summary = loop.model_summary(summary_context(module))
    assert all(is_lazy(parameter) for parameter in module.parameters())
    waiting = [hook for layer in module for hook in layer._forward_pre_hooks.values() if hasattr(hook, "done")]
    assert len(waiting) == 2 and not any(hook.done for hook in waiting)
    found = summary["net"]
    assert found["parameters"] == found["trainable"] == 13 and found["outputs"] == {"net": [4, 1]}
    assert [node["parameters"] for node in of_kind(found["graph"], "module")] == [9, 0, 4]
    assert [child["parameters"] for child in found["tree"]] == [9, 0, 4]


def test_a_lazy_model_that_cannot_run_keeps_a_summary_and_so_do_the_others():
    ctx = summary_context(nn.Linear(2, 1))
    head = nn.Sequential(nn.LazyLinear(4), nn.ReLU())
    ctx.models["head"] = ModelEntry("head", head, ["h"], ["head"])
    ctx.models["odd"] = ModelEntry("odd", Odd(), ["x"], ["odd"])
    summary = loop.model_summary(ctx)
    assert summary["net"]["graph"] is not None and summary["net"]["parameters"] == 3
    found = summary["head"]
    assert found["graph"] is None and "needs the batch field 'h'" in found["graph_note"]
    assert found["parameters"] == 0 and [child["parameters"] for child in found["tree"]] == [0, 0]
    assert all(is_lazy(parameter) for parameter in head.parameters())
    assert summary["odd"]["graph"] is None and "no parameters to count" in summary["odd"]["graph_note"]
    assert summary["odd"]["class"].endswith("Odd")
    json.dumps(summary, allow_nan=False)


def lazy_config(root):
    return {"seed": 3, "device": "cpu",
            "data": {"use": "table", "file": str(root / "housing.parquet"), "split": [0.7, 0.15, 0.15],
                     "features": {"x*": "standard_scaler"}, "targets": {"price": "standard_scaler"}, "batch": 32},
            "models": {"net": {"use": "mlp", "hidden": [8], "out": 1}},
            "losses": {"mse": "mse"}, "metrics": {"rmse": "rmse"},
            "optimizers": {"main": {"use": "adam", "lr": 0.01, "over": "@models.net", "minimizes": "@losses.mse"}},
            "train": {"epochs": 2}}


def numbers_of(result):
    return [{key: value for key, value in line.items() if key != "seconds"} for line in result.history]


@pytest.mark.slow
def test_the_start_summary_leaves_a_lazy_model_trained_the_same(tmp_path, monkeypatch):
    housing_frame(120, 0).to_parquet(tmp_path / "housing.parquet", index=False)
    summarized = kalfa.run(lazy_config(tmp_path), record=str(tmp_path / "with"))
    summary = json.loads((tmp_path / "with" / "model_summary.json").read_text(encoding="utf-8"))
    assert summary["net"]["graph"] is not None and summary["net"]["parameters"] > 0
    monkeypatch.setattr(loop, "write_summary", lambda ctx: None)
    plain = kalfa.run(lazy_config(tmp_path), record=str(tmp_path / "without"))
    assert numbers_of(summarized) == numbers_of(plain)
