import re

import numpy
import pandas
import pytest
import torch
from cirak.registry import registry

from helpers import build, frame
from kalfa.std import STD_URIS
from kalfa.std.feed.base import Dataset, IterableDataset
from kalfa.std.feed.kalfa.next_token import TokenDataset
from kalfa.std.feed.kalfa.table import SampleDataset, StreamDataset, TableDataset
from kalfa.std.feed.kalfa.window import WindowDataset
from kalfa.std.pre.base import SampleFrame, StreamFrame, TableFrame


FEEDS = ["/feed/kalfa/next_token", "/feed/kalfa/table", "/feed/kalfa/window"]


def series_frame(rows=12, sites=None, set_name="train", start=0):
    index = range(start, start + rows)
    data = pandas.DataFrame({"x0": numpy.arange(start, start + rows, dtype="float32"),
                             "x1": numpy.arange(start, start + rows, dtype="float32") * 10.0,
                             "load": numpy.arange(start, start + rows, dtype="float32") * 100.0}, index=index)
    extra = None if sites is None else pandas.DataFrame({"site": list(sites)}, index=index)
    return TableFrame(["x0", "x1"], {"load": ["load"]}, set_name, data=data, extra=extra)


def stream_frame(root, set_name="train", chunk=700, window=None):
    stream = build("/source/kalfa/parquet_stream", path=str(root / "housing.parquet"), chunk=chunk)
    if window is not None:
        stream = stream.window(*window)
    prep = build("/lego/kalfa/fit", df=stream, fields={"x0": {}, "x1": {}, "price": {"target": True}},
                 preprocessors={}, drop=[])
    return build("/lego/kalfa/apply", df=stream, prep=prep, set=set_name)


def image_frame(root, positions=None, set_name="train"):
    images = build("/source/kalfa/image_folder", path=str(root / "images"))
    if positions is not None:
        images = images.subset(positions)
    prep = build("/lego/kalfa/fit", df=images, fields={"image": {"preprocessors": ["to_t"]}, "label": {"target": True}},
                 preprocessors={"to_t": build("/pre/kalfa/to_tensor")}, drop=[])
    return build("/lego/kalfa/apply", df=images, prep=prep, set=set_name)


def text_frame(path):
    lines = build("/source/kalfa/text_lines", path=str(path))
    prep = build("/lego/kalfa/fit", df=lines, fields={"text": {"preprocessors": ["tok"]}},
                 preprocessors={"tok": build("/pre/kalfa/char_tokenizer")}, drop=[])
    return build("/lego/kalfa/apply", df=lines, prep=prep, set="train"), prep.tokenizer()


def test_feed_catalog_aliases_and_facts():
    assert sorted(uri for uri in STD_URIS if uri.startswith("/feed/")) == FEEDS
    aliases = registry.aliases()
    assert aliases["table"] == "/feed/kalfa/table"
    assert aliases["window"] == "/feed/kalfa/window"
    assert aliases["next_token"] == "/feed/kalfa/next_token"
    assert registry.facts("/feed/kalfa/window").refs == {"group": "column"}
    assert registry.facts("/feed/kalfa/window").get("needs_table") is True
    assert registry.facts("/feed/kalfa/table").get("needs_table") is None


def test_table_feed_makes_one_x_tensor_and_the_targets_by_name():
    table = frame(rows=16, features=3, seed=0)
    data = build("/feed/kalfa/table", frame=table, frames={"train": table})
    assert isinstance(data, TableDataset) and isinstance(data, Dataset)
    assert data.frame is table
    assert len(data) == 16
    assert (data.size(), data.count()) == (16, 16)
    assert (data.inputs, data.targets) == (["x"], ["price"])
    assert data.x.shape == (16, 3) and data.x.dtype == torch.float32
    item = data[3]
    assert set(item) == {"x", "price"}
    assert item["x"].shape == (3,) and item["x"].dtype == torch.float32
    assert item["price"].shape == () and item["price"].dtype == torch.float32
    torch.testing.assert_close(item["x"], torch.tensor(table.data[["x0", "x1", "x2"]].to_numpy()[3]))
    assert item["price"].item() == table.data["price"].iloc[3]
    numpy.testing.assert_array_equal(data.rows(), numpy.arange(16))
    torch.testing.assert_close(data.labels("price"), torch.tensor(table.data["price"].to_numpy()))


def test_table_feed_leaves_the_masked_rows_out():
    table = frame(rows=8, features=2, seed=2)
    table.data.index = range(100, 108)
    table.mask = numpy.array([True, False, True, True, False, True, True, True])
    data = build("/feed/kalfa/table", frame=table, frames={"train": table})
    assert len(data) == 6
    numpy.testing.assert_array_equal(data.rows(), [100, 102, 103, 105, 106, 107])
    torch.testing.assert_close(data.x, torch.tensor(table.data[["x0", "x1"]].to_numpy()[table.mask]))
    torch.testing.assert_close(data.labels("price"), torch.tensor(table.data["price"].to_numpy()[table.mask]))


def test_table_feed_keeps_target_width_and_dtype_and_takes_no_features():
    data = pandas.DataFrame({"x0": [1.0, 2.0, 3.0], "a": [10.0, 20.0, 30.0], "b": [1.0, 2.0, 3.0],
                             "label": numpy.array([0, 1, 1], dtype="int64")}, index=[5, 6, 7])
    wide = TableFrame(["x0"], {"y": ["a", "b"], "label": ["label"]}, "valid", data=data.astype({"x0": "float32"}))
    feed = build("/feed/kalfa/table", frame=wide, frames={"valid": wide})
    assert feed.targets == ["y", "label"]
    item = feed[1]
    assert item["y"].shape == (2,) and item["y"].tolist() == [20.0, 2.0]
    assert item["label"].dtype == torch.int64 and item["label"].item() == 1
    assert feed.labels("y").shape == (3, 2)
    assert feed.labels("label").tolist() == [0, 1, 1]
    numpy.testing.assert_array_equal(feed.rows(), [5, 6, 7])
    bare = TableFrame([], {"label": ["label"]}, "train", data=data)
    empty = build("/feed/kalfa/table", frame=bare, frames={"train": bare})
    assert empty.x.shape == (3, 0)
    assert empty[0]["x"].shape == (0,)
    assert len(empty) == 3


def test_table_feed_over_a_stream_yields_rows_in_chunk_order(root):
    frame = stream_frame(root, chunk=700)
    assert isinstance(frame, StreamFrame)
    data = build("/feed/kalfa/table", frame=frame, frames={"train": frame})
    assert isinstance(data, StreamDataset) and isinstance(data, IterableDataset)
    assert (data.inputs, data.targets) == (["x"], ["price"])
    assert data.size() is None
    assert (data.shuffle, data.buffer) == (False, 4096)
    items = list(data)
    assert len(items) == 2000
    assert set(items[0]) == {"x", "price"}
    assert items[0]["x"].shape == (2,) and items[0]["x"].dtype == torch.float32
    assert items[0]["price"].shape == ()
    housing = pandas.read_parquet(root / "housing.parquet")
    torch.testing.assert_close(items[1234]["x"], torch.tensor(housing[["x0", "x1"]].to_numpy(dtype="float32")[1234]))
    assert items[1234]["price"].item() == numpy.float32(housing["price"].iloc[1234])
    numpy.testing.assert_array_equal(data.rows(), numpy.arange(2000))
    assert data.count() == 2000
    with pytest.raises(ValueError, match=re.escape("a lazy set cannot be counted: balanced and class_weights need a "
                                                   "table source")):
        data.labels("price")


def test_table_feed_over_a_stream_shuffles_through_the_buffer(root):
    frame = stream_frame(root, chunk=700, window=(0, 300))
    data = build("/feed/kalfa/table", frame=frame, frames={"train": frame})
    data.shuffle = True
    data.buffer = 64
    torch.manual_seed(0)
    prices = [item["price"].item() for item in data]
    rows = data.rows()
    assert len(prices) == 300
    assert sorted(rows.tolist()) == list(range(300))
    assert rows.tolist() != list(range(300))
    housing = pandas.read_parquet(root / "housing.parquet")
    numpy.testing.assert_allclose(prices, housing["price"].to_numpy(dtype="float32")[rows])


def test_table_feed_over_an_empty_stream_reports_an_empty_train_set_only(root):
    train = stream_frame(root, window=(10, 10))
    data = build("/feed/kalfa/table", frame=train, frames={"train": train})
    with pytest.raises(ValueError, match=re.escape("the train stream yields no rows: the source, the window or the "
                                                   "filters leave nothing (the lazy set reports an empty train set "
                                                   "at its first pass)")):
        list(data)
    test = stream_frame(root, set_name="test", window=(10, 10))
    assert list(build("/feed/kalfa/table", frame=test, frames={"test": test})) == []


def test_table_feed_over_a_dataset_source_applies_the_chains_per_item(root):
    frame = image_frame(root, positions=[0, 1, 40])
    assert isinstance(frame, SampleFrame)
    data = build("/feed/kalfa/table", frame=frame, frames={"train": frame})
    assert isinstance(data, SampleDataset)
    assert (data.inputs, data.targets) == (["image"], ["label"])
    assert len(data) == 3
    item = data[2]
    assert set(item) == {"image", "label"}
    assert item["image"].shape == (1, 8, 8) and item["image"].dtype == torch.float32
    assert item["label"].dtype == torch.int64 and item["label"].item() == 1
    assert data[0]["label"].item() == 0
    numpy.testing.assert_array_equal(data.rows(), [0, 1, 40])
    assert data.labels("label").tolist() == [0, 0, 1]
    assert data.labels("label").dtype == torch.int64
    assert data.size() == 3
    whole = build("/feed/kalfa/table", frame=image_frame(root), frames={})
    assert len(whole) == 64
    assert whole.labels("label").sum().item() == 24


def test_window_feed_slides_size_steps_and_the_next_horizon_targets():
    table = series_frame(rows=10)
    data = build("/feed/kalfa/window", frame=table, frames={"train": table}, size=3, horizon=2)
    assert isinstance(data, WindowDataset)
    assert (data.window, data.horizon) == (3, 2)
    assert (data.inputs, data.targets) == (["x"], ["load"])
    assert len(data) == 6
    assert data.x.shape == (6, 3, 2) and data.x.dtype == torch.float32
    assert data.fields["load"].shape == (6, 2)
    first = data[0]
    assert set(first) == {"x", "load"}
    assert first["x"].tolist() == [[0.0, 0.0], [1.0, 10.0], [2.0, 20.0]]
    assert first["load"].tolist() == [300.0, 400.0]
    last = data[5]
    assert last["x"][:, 0].tolist() == [5.0, 6.0, 7.0]
    assert last["load"].tolist() == [800.0, 900.0]
    numpy.testing.assert_array_equal(data.rows(), [3, 4, 5, 6, 7, 8])
    one = build("/feed/kalfa/window", frame=table, frames={"train": table}, size=3, horizon=1)
    assert len(one) == 7
    assert one.fields["load"].shape == (7, 1)
    assert one[6]["load"].tolist() == [900.0]


def test_window_feed_keeps_wide_targets_and_is_empty_below_size_plus_horizon():
    data = pandas.DataFrame({"x0": numpy.arange(6, dtype="float32"), "a": numpy.arange(6, dtype="float32"),
                             "b": -numpy.arange(6, dtype="float32")})
    wide = TableFrame(["x0"], {"y": ["a", "b"]}, "train", data=data)
    feed = build("/feed/kalfa/window", frame=wide, frames={"train": wide}, size=2, horizon=2)
    assert len(feed) == 3
    assert feed.fields["y"].shape == (3, 2, 2)
    assert feed[0]["y"].tolist() == [[2.0, -2.0], [3.0, -3.0]]
    short = build("/feed/kalfa/window", frame=series_frame(rows=4), frames={}, size=3, horizon=2)
    assert len(short) == 0
    assert short.x.shape == (0, 3, 2)
    assert short.fields["load"].shape == (0, 2)
    assert short.rows().shape == (0,)


def test_window_feed_with_a_group_keeps_every_series_apart():
    table = series_frame(rows=12, sites=["A"] * 6 + ["B"] * 6)
    data = build("/feed/kalfa/window", frame=table, frames={"train": table}, size=2, horizon=1, group="site")
    assert len(data) == 8
    numpy.testing.assert_array_equal(data.rows(), [2, 3, 4, 5, 8, 9, 10, 11])
    assert data[3]["x"][:, 0].tolist() == [3.0, 4.0]
    assert data[3]["load"].tolist() == [500.0]
    assert data[4]["x"][:, 0].tolist() == [6.0, 7.0]
    assert data[4]["load"].tolist() == [800.0]
    mixed = series_frame(rows=6, sites=["A", "B", "A", "B", "A", "B"])
    split = build("/feed/kalfa/window", frame=mixed, frames={}, size=2, horizon=1, group="site")
    assert len(split) == 2
    assert split[0]["x"][:, 0].tolist() == [0.0, 2.0]
    assert split[1]["x"][:, 0].tolist() == [1.0, 3.0]
    numpy.testing.assert_array_equal(split.rows(), [4, 5])
    unknown = build("/feed/kalfa/window", frame=table, frames={}, size=2, horizon=1, group="nope")
    assert len(unknown) == 10
    plain = build("/feed/kalfa/window", frame=series_frame(rows=12), frames={}, size=2, horizon=1, group="site")
    assert len(plain) == 10


def test_window_feed_with_context_takes_the_tail_of_the_previous_sets():
    train = series_frame(rows=10)
    valid = series_frame(rows=4, set_name="valid", start=10)
    test = series_frame(rows=3, set_name="test", start=14)
    frames = {"train": train, "valid": valid, "test": test}
    alone = build("/feed/kalfa/window", frame=valid, frames=frames, size=3, horizon=1)
    assert len(alone) == 1
    numpy.testing.assert_array_equal(alone.rows(), [13])
    joined = build("/feed/kalfa/window", frame=valid, frames=frames, size=3, horizon=1, context=True)
    assert len(joined) == 4
    numpy.testing.assert_array_equal(joined.rows(), [10, 11, 12, 13])
    assert joined[0]["x"][:, 0].tolist() == [7.0, 8.0, 9.0]
    assert joined[0]["load"].tolist() == [1000.0]
    assert joined[3]["x"][:, 0].tolist() == [10.0, 11.0, 12.0]
    assert joined[3]["load"].tolist() == [1300.0]
    tail = build("/feed/kalfa/window", frame=test, frames=frames, size=3, horizon=1, context=True)
    assert len(tail) == 3
    assert tail[0]["x"][:, 0].tolist() == [11.0, 12.0, 13.0]
    assert tail[0]["load"].tolist() == [1400.0]
    numpy.testing.assert_array_equal(tail.rows(), [14, 15, 16])
    first = build("/feed/kalfa/window", frame=train, frames=frames, size=3, horizon=1, context=True)
    assert len(first) == 7
    hollow = {"train": train, "valid": series_frame(rows=0, set_name="valid", start=10), "test": test}
    skipped = build("/feed/kalfa/window", frame=test, frames=hollow, size=3, horizon=1, context=True)
    assert len(skipped) == 3
    assert skipped[0]["x"][:, 0].tolist() == [7.0, 8.0, 9.0]


def test_window_feed_with_context_and_group_keeps_the_tails_per_series():
    train = series_frame(rows=6, sites=["A", "B"] * 3)
    valid = series_frame(rows=4, sites=["A", "B", "A", "B"], set_name="valid", start=6)
    data = build("/feed/kalfa/window", frame=valid, frames={"train": train, "valid": valid}, size=2, horizon=1,
                 context=True, group="site")
    assert len(data) == 4
    numpy.testing.assert_array_equal(data.rows(), [6, 8, 7, 9])
    assert data[0]["x"][:, 0].tolist() == [2.0, 4.0]
    assert data[0]["load"].tolist() == [600.0]
    assert data[2]["x"][:, 0].tolist() == [3.0, 5.0]
    assert data[2]["load"].tolist() == [700.0]


def test_window_feed_refuses_a_stream(root):
    frame = stream_frame(root)
    with pytest.raises(ValueError, match=re.escape("window needs a table in memory; the lazy set has the table feed "
                                                   "only")):
        build("/feed/kalfa/window", frame=frame, frames={"train": frame}, size=3, horizon=1)


def test_next_token_feed_windows_the_token_stream_with_shifted_targets(tmp_path):
    (tmp_path / "tiny.txt").write_text("ab\nc\n", encoding="utf-8")
    frame, tokenizer = text_frame(tmp_path / "tiny.txt")
    assert tokenizer.chars == ["\n", "a", "b", "c"]
    data = build("/feed/kalfa/next_token", frame=frame, frames={"train": frame}, seq_len=2)
    assert isinstance(data, TokenDataset)
    assert (data.inputs, data.targets, data.seq_len) == (["input_ids"], ["targets"], 2)
    assert len(data) == 2
    assert data.input_ids.tolist() == [[1, 2], [0, 3]]
    assert data.target_ids.tolist() == [[2, 0], [3, 0]]
    item = data[1]
    assert set(item) == {"input_ids", "targets"}
    assert item["input_ids"].dtype == torch.int64 and item["targets"].dtype == torch.int64
    assert item["input_ids"].tolist() == [0, 3]
    numpy.testing.assert_array_equal(data.rows(), [0, 2])
    assert data.labels("targets") is data.target_ids
    assert data.size() == 2
    four = build("/feed/kalfa/next_token", frame=frame, frames={"train": frame}, seq_len=4)
    assert four.input_ids.tolist() == [[1, 2, 0, 3]]
    assert four.target_ids.tolist() == [[2, 0, 3, 0]]
    none = build("/feed/kalfa/next_token", frame=frame, frames={"train": frame}, seq_len=5)
    assert len(none) == 0
    assert none.input_ids.shape == (0, 5) and none.target_ids.shape == (0, 5)
    assert none.rows().shape == (0,)


def test_next_token_feed_over_a_text_file_decodes_back_to_its_start(root):
    frame, tokenizer = text_frame(root / "text.txt")
    data = build("/feed/kalfa/next_token", frame=frame, frames={"train": frame}, seq_len=32)
    text = (root / "text.txt").read_text(encoding="utf-8")
    total = sum(len(line) + 1 for line in text.splitlines())
    assert len(data) == (total - 1) // 32
    assert tokenizer.decode(data[0]["input_ids"]) == text[:32]
    assert tokenizer.decode(data[0]["targets"]) == text[1:33]
    assert tokenizer.decode(data[1]["input_ids"]) == text[32:64]
    numpy.testing.assert_array_equal(data.rows(), numpy.arange(len(data)) * 32)
