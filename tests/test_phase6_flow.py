import json
from pathlib import Path
from types import SimpleNamespace

import numpy
import pytest
from PIL import Image

import kalfa
from helpers import housing_frame
from kalfa.board.answers import Board
from kalfa.core.rng import Rng
from kalfa.records import reader
from kalfa.std.data import flow
from kalfa.std.data.images import ImageFolder
from kalfa.std.data.table import Table
from kalfa.std.data.text import TextLines


LINES = ["the quick brown fox", "jumps over the lazy dog", "a stitch in time", "saves nine", "all that glitters",
         "is not gold", "the early bird", "catches the worm", "actions speak louder", "than words",
         "practice makes perfect", "time flies like an arrow", "fruit flies like a banana", "better late than never"]


class Plain:
    inputs = ("x",)
    target_names = ["y"]

    def report(self):
        return {"rows": {"train": 10, "valid": 2}}


def make_ctx(directory, seed=7):
    return SimpleNamespace(seed=seed, rng=Rng(seed), config=SimpleNamespace(directories=[str(directory)]))


def image_tree(root, counts=(8, 8, 8)):
    for position, (name, count) in enumerate(zip(("cat", "dog", "owl"), counts)):
        for index in range(count):
            path = root / name / f"{index:02d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            array = numpy.zeros((8, 8, 3), dtype="uint8")
            array[:, :4] = 40 + 60 * position + index
            Image.fromarray(array).save(path)
    return root


def following(pipeline):
    found = {}
    for edge in pipeline["edges"]:
        found.setdefault(edge["from"], []).append(edge["to"])
    return found


def reachable(pipeline, start):
    nexts, seen, todo = following(pipeline), set(), [start]
    while todo:
        node = todo.pop()
        if node not in seen:
            seen.add(node)
            todo.extend(nexts.get(node, []))
    return seen


def acyclic(pipeline):
    nexts = following(pipeline)
    incoming = {node["id"]: 0 for node in pipeline["nodes"]}
    for edge in pipeline["edges"]:
        incoming[edge["to"]] += 1
    ready = [key for key, count in incoming.items() if count == 0]
    done = 0
    while ready:
        node = ready.pop()
        done += 1
        for child in nexts.get(node, []):
            incoming[child] -= 1
            if incoming[child] == 0:
                ready.append(child)
    return done == len(incoming)


def checked_shape(pipeline, data):
    json.dumps(pipeline, allow_nan=False)
    nodes = {node["id"]: node for node in pipeline["nodes"]}
    assert len(nodes) == len(pipeline["nodes"])
    for node in nodes.values():
        assert {"id", "kind", "label", "detail", "count", "group"} <= set(node), node
    assert all(edge["from"] in nodes and edge["to"] in nodes for edge in pipeline["edges"])
    assert acyclic(pipeline)
    sources = [key for key, node in nodes.items() if node["kind"] == "source"]
    assert sources
    assert set().union(*(reachable(pipeline, key) for key in sources)) == set(nodes)
    [batch] = [key for key, node in nodes.items() if node["kind"] == "batch"]
    for key, node in nodes.items():
        if node["kind"] in ("set", "tensor"):
            assert batch in reachable(pipeline, key), node
    sets = {node["label"]: node["count"] for node in nodes.values() if node["kind"] == "set"}
    assert sets == data.report()["rows"]
    groups = {group["id"] for group in pipeline.get("groups") or []}
    assert all(node["group"] is None or node["group"] in groups for node in nodes.values())
    return nodes


def kinds_of(nodes):
    return {node["kind"] for node in nodes.values()}


def labels_of(nodes, kind):
    return [node["label"] for node in nodes.values() if node["kind"] == kind]


def test_the_table_pipeline_has_a_row_lane_and_a_column_lane(tmp_path):
    frame = housing_frame(200, 0)
    frame["note"] = "n"
    frame.to_parquet(tmp_path / "housing.parquet", index=False)
    data = Table("housing.parquet", split=[0.7, 0.15, 0.15], features={"x*": "standard_scaler"},
                 targets={"price": "standard_scaler"}, filter="price > 0", drop=["note"], batch=32)
    data.fit(make_ctx(tmp_path))
    nodes = checked_shape(flow.pipeline_of(data), data)
    assert {"source", "filter", "drop", "split", "set", "features", "chain", "targets", "tensor",
            "batch"} <= kinds_of(nodes)
    assert set(labels_of(nodes, "tensor")) == {"x", "price"}
    assert [node["count"] for node in nodes.values() if node["kind"] == "features"] == [4]
    assert labels_of(nodes, "chain").count("standard_scaler") == 2


def test_the_image_folder_pipeline_shows_classes_and_both_transform_chains(tmp_path):
    root = image_tree(tmp_path / "pets")
    data = ImageFolder(str(root), split=[0.5, 0.25, 0.25], batch=4, transform=["to_tensor"],
                       train_transform=[{"use": "random_flip", "p": 0.5}, "to_tensor"])
    data.fit(make_ctx(tmp_path))
    nodes = checked_shape(flow.pipeline_of(data), data)
    assert {"source", "classes", "split", "set", "transform", "tensor", "batch"} <= kinds_of(nodes)
    [classes] = [node for node in nodes.values() if node["kind"] == "classes"]
    assert classes["count"] == 3
    assert "random_flip" in labels_of(nodes, "transform") and labels_of(nodes, "transform").count("to_tensor") == 2
    assert set(labels_of(nodes, "tensor")) == {"image", "label"}


def test_the_text_pipeline_shows_the_tokenizer_and_the_windows(tmp_path):
    (tmp_path / "corpus.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    data = TextLines("corpus.txt", seq_len=8, split={"use": "sequential", "ratios": [0.6, 0.2, 0.2]}, batch=4)
    data.fit(make_ctx(tmp_path))
    nodes = checked_shape(flow.pipeline_of(data), data)
    assert {"source", "split", "set", "tokenizer", "windows", "tensor", "batch"} <= kinds_of(nodes)
    [tokenizer] = [node for node in nodes.values() if node["kind"] == "tokenizer"]
    assert tokenizer["count"] == data.vocab_size
    assert "input_ids" in labels_of(nodes, "tensor")


def test_a_data_object_without_a_pipeline_gets_a_generic_one():
    nodes = checked_shape(flow.pipeline_of(Plain()), Plain())
    assert set(labels_of(nodes, "tensor")) == {"x", "y"}


def test_a_run_writes_data_json_and_the_board_serves_it(tmp_path):
    housing_frame(120, 0).to_parquet(tmp_path / "housing.parquet", index=False)
    config = {
        "seed": 3, "device": "cpu",
        "data": {"use": "table", "file": str(tmp_path / "housing.parquet"), "split": [0.7, 0.15, 0.15],
                 "features": {"x*": "standard_scaler"}, "targets": {"price": "standard_scaler"}, "batch": 32},
        "models": {"net": {"use": "mlp", "hidden": [8], "out": 1}},
        "losses": {"mse": "mse"},
        "optimizers": {"main": {"use": "adam", "lr": 0.01, "over": "@models.net", "minimizes": "@losses.mse"}},
        "train": {"epochs": 1},
    }
    result = kalfa.run(config, record=str(tmp_path / "runs" / "net"))
    found = json.loads((Path(result.record) / "data.json").read_text(encoding="utf-8"))
    assert found["format"] == 1 and found["kind"] == "table"
    assert found["report"]["features"] == 4 and sum(found["report"]["rows"].values()) == 120
    assert {node["kind"] for node in found["pipeline"]["nodes"]} >= {"source", "split", "set", "tensor", "batch"}
    assert reader.data_report_of(result.record) == found
    assert kalfa.open(result.record).data_report == found
    assert Board(tmp_path / "runs").record("net")["data_report"]["pipeline"] == found["pipeline"]
