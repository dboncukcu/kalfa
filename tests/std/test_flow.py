import json
import logging
from types import SimpleNamespace

import numpy
import pandas
import pytest
from PIL import Image

import kalfa
from helpers import blobs_frame, housing_frame
from kalfa.core import run as running
from kalfa.core.rng import Rng
from kalfa.std.data import flow
from kalfa.std.data.images import ImageFolder
from kalfa.std.data.table import Table
from kalfa.std.data.text import TextLines
from kalfa.std.data.window import Window


LINES = ["the quick brown fox", "jumps over the lazy dog", "a stitch in time", "saves nine", "all that glitters",
         "is not gold", "the early bird", "catches the worm", "actions speak louder", "than words"]


def make_ctx(directory, seed=7):
    return SimpleNamespace(seed=seed, rng=Rng(seed), config=SimpleNamespace(directories=[str(directory)]))


def reachable(pipeline, start):
    nexts = {}
    for edge in pipeline["edges"]:
        nexts.setdefault(edge["from"], []).append(edge["to"])
    seen, todo = set(), [start]
    while todo:
        node = todo.pop()
        if node not in seen:
            seen.add(node)
            todo.extend(nexts.get(node, []))
    return seen


def checked(pipeline, data):
    json.dumps(pipeline, allow_nan=False)
    nodes = {node["id"]: node for node in pipeline["nodes"]}
    assert len(nodes) == len(pipeline["nodes"])
    assert all(edge["from"] in nodes and edge["to"] in nodes for edge in pipeline["edges"])
    [source] = [key for key, node in nodes.items() if node["kind"] == "source"]
    assert reachable(pipeline, source) == set(nodes)
    [batch] = [key for key, node in nodes.items() if node["kind"] == "batch"]
    for key, node in nodes.items():
        if node["kind"] in ("set", "tensor"):
            assert batch in reachable(pipeline, key), node
    assert {node["label"]: node["count"] for node in nodes.values() if node["kind"] == "set"} == data.report()["rows"]
    groups = {group["id"] for group in pipeline["groups"]}
    assert all(node["group"] is None or node["group"] in groups for node in nodes.values())
    return nodes


def of_kind(nodes, kind):
    return [node for node in nodes.values() if node["kind"] == kind]


def edges_into(pipeline, key):
    return {edge["from"]: edge["label"] for edge in pipeline["edges"] if edge["to"] == key}


def test_filters_per_set_show_on_their_set_and_the_rows_read_on_the_source(tmp_path):
    housing_frame(200, 0).to_parquet(tmp_path / "housing.parquet", index=False)
    data = Table("housing.parquet", split=[0.6, 0.2, 0.2],
                 features={"x*": ["minmax_scaler", {"use": "log", "base": 2}]}, targets={"price": None},
                 filter={"all": "x0 > -1.5", "test": "x1 > -1"}, batch=16)
    data.fit(make_ctx(tmp_path))
    pipeline = flow.pipeline_of(data)
    nodes = checked(pipeline, data)
    [source] = of_kind(nodes, "source")
    [rows] = of_kind(nodes, "filter")
    assert source["count"] == 200 and source["label"] == "housing.parquet"
    assert rows["detail"] == "x0 > -1.5" and rows["count"] == data.kept_rows < 200
    sets = {node["label"]: node for node in of_kind(nodes, "set")}
    assert "x1 > -1" in sets["test"]["detail"] and sets["train"]["detail"] == ""
    assert edges_into(pipeline, "set:train") == {"split": "60%"}
    assert [(node["label"], node["detail"]) for node in of_kind(nodes, "chain")] == [("minmax_scaler", ""),
                                                                                   ("log", "base 2")]
    [targets] = of_kind(nodes, "targets")
    assert targets["label"] == "price" and [group["label"] for group in pipeline["groups"]] == ["x*"]


def test_a_kfold_split_names_the_fold_and_a_label_target_its_classes(tmp_path):
    blobs_frame(240).to_parquet(tmp_path / "blobs.parquet", index=False)
    data = Table("blobs.parquet", split={"use": "kfold", "folds": 4, "fold": 1, "valid_ratio": 0.25},
                 features={"f*": "standard_scaler"}, targets={"kind": "label"}, drop=["id"], batch=32)
    data.fit(make_ctx(tmp_path))
    pipeline = flow.pipeline_of(data)
    nodes = checked(pipeline, data)
    [split] = of_kind(nodes, "split")
    assert split["label"] == "kfold split" and "fold 1 of 4" in split["detail"]
    assert edges_into(pipeline, "set:test") == {"split": "fold 1"}
    assert edges_into(pipeline, "set:train") == {"split": "other folds"}
    [drop] = of_kind(nodes, "drop")
    assert drop["detail"] == "id" and drop["count"] == 1
    [targets] = of_kind(nodes, "targets")
    assert targets["count"] == 3 and targets["detail"] == "cold, hot, mild"
    tensors = {node["label"]: node["detail"] for node in of_kind(nodes, "tensor")}
    assert tensors == {"x": "2 features", "kind": "3 classes"}


def test_a_window_feed_gathers_both_lanes_before_the_batch(tmp_path):
    generator = numpy.random.default_rng(0)
    frame = pandas.DataFrame({"g": numpy.repeat(["a", "b"], 60), "f1": generator.normal(size=120),
                              "f2": generator.normal(size=120), "y": generator.normal(size=120)})
    frame.to_parquet(tmp_path / "series.parquet", index=False)
    data = Table("series.parquet", split={"use": "sequential", "ratios": [0.6, 0.2, 0.2], "group": "g"},
                 features={"f*": None}, targets={"y": "standard_scaler"}, feed=Window(5, horizon=2, group="g"),
                 batch=16)
    data.fit(make_ctx(tmp_path))
    pipeline = flow.pipeline_of(data)
    nodes = checked(pipeline, data)
    [feed] = of_kind(nodes, "feed")
    assert feed["label"] == "window" and feed["count"] == sum(data.windows().values())
    assert "size 5" in feed["detail"] and "horizon 2" in feed["detail"]
    assert set(edges_into(pipeline, feed["id"])) == {"set:train", "set:valid", "set:test", "tensor:x", "tensor:y"}
    assert set(edges_into(pipeline, "batch")) == {feed["id"]}
    assert [node["label"] for node in of_kind(nodes, "chain")] == ["standard_scaler"]
    sets = {node["label"]: node["detail"] for node in of_kind(nodes, "set")}
    assert sets["train"] == f"{data.windows()['train']} windows"
    [split] = of_kind(nodes, "split")
    assert split["detail"] == "60%, 20%, 20%, in file order, within each g"


def test_a_stream_table_reads_in_chunks_and_filters_after_the_split(tmp_path):
    generator = numpy.random.default_rng(1)
    frame = pandas.DataFrame({"x1": generator.normal(size=300), "x2": generator.normal(size=300),
                              "y": generator.normal(size=300)})
    frame.to_parquet(tmp_path / "rows.parquet", index=False, row_group_size=50)
    data = Table("rows.parquet", split={"use": "sequential", "ratios": [0.7, 0.15, 0.15]},
                 features={"x*": "standard_scaler"}, targets={"y": None}, filter="x1 > -2", batch=32,
                 stream={"chunk": 64})
    data.fit(make_ctx(tmp_path))
    pipeline = flow.pipeline_of(data)
    nodes = checked(pipeline, data)
    [stream] = of_kind(nodes, "stream")
    assert "chunks of 64 rows" in stream["detail"]
    [source] = of_kind(nodes, "source")
    assert source["count"] == 300
    assert edges_into(pipeline, "stream") == {"source": ""} and edges_into(pipeline, "split") == {"stream": ""}
    assert edges_into(pipeline, "filter") == {"split": ""}
    sets = {node["label"]: node["detail"] for node in of_kind(nodes, "set")}
    assert sets["train"] == "streamed" and sets["valid"] == ""


def test_a_record_reloaded_for_predict_draws_its_pipeline(tmp_path):
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
    written = json.loads((tmp_path / "runs" / "net" / "data.json").read_text(encoding="utf-8"))
    loaded = Table(str(tmp_path / "housing.parquet"), split=[0.7, 0.15, 0.15], features={"x*": "standard_scaler"},
                   targets={"price": "standard_scaler"}, batch=32).load(result.record)
    bare = flow.pipeline_of(loaded)
    nodes = checked(bare, loaded)
    assert not of_kind(nodes, "set") and of_kind(nodes, "source")[0]["count"] is None
    assert [node["count"] for node in of_kind(nodes, "features")] == [4]
    loaded.rebuild(make_ctx(tmp_path, seed=3))
    rebuilt = flow.pipeline_of(loaded)
    checked(rebuilt, loaded)
    assert rebuilt == written["pipeline"]


def test_a_reloaded_image_folder_and_text_file_keep_their_shape(tmp_path):
    for position, name in enumerate(("cat", "dog")):
        for index in range(6):
            path = tmp_path / "pets" / name / f"{index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(numpy.full((8, 8, 3), 40 * position + index, dtype="uint8")).save(path)
    images = ImageFolder("pets", split=[0.5, 0.5], transform=["to_tensor"], batch=4)
    images.fit(make_ctx(tmp_path))
    images.save(tmp_path / "record")
    again = ImageFolder("pets", split=[0.5, 0.5], transform=["to_tensor"], batch=4).load(tmp_path / "record")
    nodes = checked(flow.pipeline_of(again), again)
    assert of_kind(nodes, "classes")[0]["count"] == 2 and of_kind(nodes, "source")[0]["count"] is None
    assert [node["label"] for node in of_kind(nodes, "transform")] == ["to_tensor"]
    again.rebuild(make_ctx(tmp_path))
    nodes = checked(flow.pipeline_of(again), again)
    assert of_kind(nodes, "source")[0]["count"] == 12
    (tmp_path / "corpus.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    text = TextLines("corpus.txt", seq_len=8, split={"use": "sequential", "ratios": [0.8, 0.2]}, batch=4)
    text.fit(make_ctx(tmp_path))
    text.save(tmp_path / "record")
    reread = TextLines("corpus.txt", seq_len=8, split={"use": "sequential", "ratios": [0.8, 0.2]})
    reread.load(tmp_path / "record")
    nodes = checked(flow.pipeline_of(reread), reread)
    [tokenizer] = of_kind(nodes, "tokenizer")
    assert tokenizer["count"] == text.vocab_size and of_kind(nodes, "windows")[0]["count"] is None


def test_a_text_set_too_short_for_a_window_shows_as_left_out(tmp_path):
    (tmp_path / "corpus.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    data = TextLines("corpus.txt", seq_len=40, split={"use": "sequential", "ratios": [0.9, 0.1]}, batch=4)
    data.fit(make_ctx(tmp_path))
    nodes = checked(flow.pipeline_of(data), data)
    [dropped] = of_kind(nodes, "drop")
    assert dropped["label"] == "valid left out" and "fewer than seq_len + 1 = 41" in dropped["detail"]
    [train] = of_kind(nodes, "set")
    assert "lines" in train["detail"] and "tokens" in train["detail"]


def test_kind_is_the_alias_or_the_class_path_and_values_become_strict_json():
    assert flow.kind_of(Table("x.parquet", split=[0.8, 0.2])) == "table"
    assert flow.kind_of(SimpleNamespace()) == "types.SimpleNamespace"
    found = flow.jsonable({"a": numpy.int64(3), "b": numpy.float32(0.5), "c": float("nan"), "d": numpy.arange(2),
                           1: (numpy.bool_(True),)})
    assert found == {"a": 3, "b": 0.5, "c": None, "d": [0, 1], "1": [True]}
    json.dumps(found, allow_nan=False)


class Broken:
    inputs = ("x",)

    def report(self):
        return {"rows": {"train": 4}}

    def pipeline(self):
        raise ValueError("no picture")


def test_a_failing_pipeline_writes_null_and_warns(caplog):
    ctx = SimpleNamespace(data=Broken(), logger=logging.getLogger("kalfa.run"))
    with caplog.at_level("WARNING", logger="kalfa.run"):
        document = running.data_document(ctx)
    assert document == {"format": 1, "kind": f"{__name__}.Broken", "report": {"rows": {"train": 4}}, "pipeline": None}
    assert "no picture" in caplog.text


def test_resume_writes_data_json_again(tmp_path):
    housing_frame(120, 0).to_parquet(tmp_path / "housing.parquet", index=False)
    config = {
        "seed": 3, "device": "cpu",
        "data": {"use": "table", "file": str(tmp_path / "housing.parquet"), "split": [0.7, 0.15, 0.15],
                 "features": {"x*": "standard_scaler"}, "targets": {"price": None}, "batch": 32},
        "models": {"net": {"use": "mlp", "hidden": [8], "out": 1}},
        "losses": {"mse": "mse"},
        "optimizers": {"main": {"use": "adam", "lr": 0.01, "over": "@models.net", "minimizes": "@losses.mse"}},
        "train": {"epochs": 1},
    }
    first = kalfa.run(config, record=str(tmp_path / "runs" / "one"))
    second = running.resume(first.record, sets=[("train.epochs", 2)], target=tmp_path / "runs" / "two")
    before = json.loads((tmp_path / "runs" / "one" / "data.json").read_text(encoding="utf-8"))
    after = json.loads((tmp_path / "runs" / "two" / "data.json").read_text(encoding="utf-8"))
    assert second.record.endswith("two") and after == before


@pytest.mark.parametrize("kind", ["filter", "drop", "stream", "feed"])
def test_optional_nodes_stay_out_when_unused(tmp_path, kind):
    housing_frame(60, 0).to_parquet(tmp_path / "housing.parquet", index=False)
    data = Table("housing.parquet", split=[0.8, 0.2], features={"x*": None}, targets={"price": None}, batch=8)
    data.fit(make_ctx(tmp_path))
    nodes = checked(flow.pipeline_of(data), data)
    assert not of_kind(nodes, kind)
