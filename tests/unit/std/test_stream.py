"""The lazy set: chunked sources, filters and windows on the stream, streamed fit, the stream dataset and loader."""

import functools

import numpy
import pandas
import pytest
import torch

import kalfa  # noqa: F401
from kalfa.std.common.stream import positions
from kalfa.std.feed.kalfa.table import StreamDataset, table
from kalfa.std.feed.kalfa.window import window
from kalfa.std.lego.kalfa.data_steps import transform_set
from kalfa.std.lego.kalfa.headers import csv_header, parquet_header
from kalfa.std.lego.kalfa.prep import apply, fit
from kalfa.std.loader.kalfa.torch import torch_loader
from kalfa.std.pre.kalfa.encoders import OneHot
from kalfa.std.pre.sklearn.scalers import StandardScaler
from kalfa.std.source.kalfa.tables import csv_stream, parquet_stream
from kalfa.std.split.kalfa.splits import given, kfold, random_split, sequential
from kalfa.std.transform.kalfa.table import derive, filter_rows
from kalfa.synthetic import housing_frame


@pytest.fixture
def housing(tmp_path):
    frame = housing_frame(rows=100, seed=0, columns=3)
    frame["kind"] = numpy.where(numpy.arange(100) % 4 == 0, "odd", "even")
    frame.to_parquet(tmp_path / "housing.parquet", index=False)
    frame.to_csv(tmp_path / "housing.csv", index=False)
    return tmp_path, frame


def test_chunked_sources_yield_the_table_with_global_row_ids(housing):
    directory, frame = housing
    stream = parquet_stream(str(directory / "housing.parquet"), chunk=30)
    chunks = list(stream.chunks())
    assert [len(chunk) for chunk in chunks] == [30, 30, 30, 10] and stream.rows == 100
    joined = pandas.concat(chunks)
    assert joined.index.tolist() == list(range(100)) and list(joined.columns) == list(frame.columns)
    assert parquet_header(str(directory / "housing.parquet"))["rows"] == 100
    csv = csv_stream(str(directory / "housing.csv"), chunk=40)
    assert positions(csv).tolist() == list(range(100)) and csv.columns == list(frame.columns)
    assert csv_header(str(directory / "housing.csv"))["rows"] == 100


def test_filters_and_windows_apply_on_the_stream(housing):
    directory, frame = housing
    stream = parquet_stream(str(directory / "housing.parquet"), chunk=30)
    odd = filter_rows(stream, "kind == 'odd'")
    assert positions(odd).tolist() == list(range(0, 100, 4)) and odd.count() == 25
    parts = sequential(stream, [0.7, 0.2, 0.1])
    assert [part.rows for part in parts.values()] == [70, 20, 10]
    assert positions(parts["valid"]).tolist() == list(range(70, 90))
    expensive = [functools.partial(filter_rows, query="price > 200")]
    assert positions(transform_set(parts["train"], "train", expensive)).tolist() == \
        [position for position in range(70) if frame["price"][position] > 200]
    assert positions(transform_set(parts["test"], "test", [])).tolist() == list(range(90, 100))
    with pytest.raises(ValueError, match="needs a table"):
        derive(stream, "twice", "price * 2")
    with pytest.raises(ValueError, match="sequential or given"):
        random_split(stream, [0.7, 0.2, 0.1], seed=1)
    with pytest.raises(ValueError, match="sequential or given"):
        kfold(stream, k=5, fold=0)
    with pytest.raises(ValueError, match="group"):
        sequential(stream, [0.7, 0.2, 0.1], group="kind")
    parts = given(stream, test=str(directory / "housing.parquet"))
    assert parts["test"].rows == 100 and parts["valid"].rows == 0 and parts["train"] is stream


def test_streamed_fit_matches_the_table_fit(housing):
    directory, frame = housing
    stream = parquet_stream(str(directory / "housing.parquet"), chunk=30)
    fields = {"x*": {"preprocessors": ["scale"]}, "kind": {"preprocessors": ["hot"]}, "price": {"target": True}}
    pres = {"scale": StandardScaler(), "hot": OneHot()}
    lazy = fit(stream, fields, pres, [])
    eager = fit(frame, fields, {"scale": StandardScaler(), "hot": OneHot()}, [])
    assert lazy.features == eager.features and lazy.dtypes == eager.dtypes
    values = numpy.linspace(-2.0, 2.0, 5)
    for name in ("x0", "x1", "x2"):
        # the lazy set fits a grouped preprocessor column by column, the table fit fits one object over the block;
        # for a per column scaler the two are the same transform
        assert lazy.object_of("scale", name).apply(values) == pytest.approx(
            eager.object_of("scale", name).apply(values))
    assert lazy.object_of("hot", "kind").columns("kind") == eager.object_of("hot", "kind").columns("kind")
    view = apply(stream, lazy, "test")
    assert view.stream is not None and view.index is None
    with pytest.raises(TypeError, match="no length"):
        len(view)
    chunks = list(view.stream.chunks())
    joined = pandas.concat([data for _, data in chunks])
    reference = apply(frame, eager, "test").data
    assert list(joined.columns) == list(reference.columns)
    assert numpy.allclose(joined.to_numpy(dtype="float64"), reference.to_numpy(dtype="float64"))


def test_stream_dataset_and_loader(housing):
    directory, frame = housing
    stream = parquet_stream(str(directory / "housing.parquet"), chunk=30)
    fields = {"x*": {"preprocessors": ["scale"]}, "price": {"target": True}}
    prep = fit(stream, fields, {"scale": StandardScaler()}, ["kind"])
    parts = sequential(stream, [0.7, 0.2, 0.1])
    train = table(apply(parts["train"], prep, "train"))
    assert isinstance(train, StreamDataset) and train.size() is None and train.count() == 70
    loader = torch_loader(train, "train", 16, buffer=8)
    torch.manual_seed(3)
    first = [batch["x"] for batch in loader]
    order_first = train.rows().tolist()
    torch.manual_seed(3)
    list(loader)
    assert train.rows().tolist() == order_first and sorted(order_first) == list(range(70))
    assert order_first != list(range(70)) and first[0].shape == (16, 3)
    test = torch_loader(table(apply(parts["test"], prep, "test")), "test", 16)
    batches = list(test)
    assert test.dataset.rows().tolist() == list(range(90, 100)) and batches[0]["price"].shape == (10,)
    with pytest.raises(ValueError, match="cannot be counted"):
        torch_loader(train, "train", 16, balanced=True)
    with pytest.raises(ValueError, match="workers: 0"):
        torch_loader(train, "train", 16, workers=2)
    with pytest.raises(ValueError, match="cannot be counted"):
        train.labels("price")
    with pytest.raises(ValueError, match="table in memory"):
        window(apply(parts["train"], prep, "train"), None, size=2, horizon=1)
    empty = table(apply(filter_rows(parts["train"], "price > 1e9"), prep, "train"))
    with pytest.raises(ValueError, match="yields no rows"):
        list(torch_loader(empty, "train", 4))
    assert list(torch_loader(table(apply(parts["train"].empty(), prep, "valid")), "valid", 4)) == []


def test_a_stream_reads_the_listed_columns(housing):
    directory, _ = housing
    stream = parquet_stream(str(directory / "housing.parquet"), chunk=30, columns=["price"])
    assert all(list(frame.columns) == ["price"] for frame in stream.chunks())
    stream = csv_stream(str(directory / "housing.csv"), chunk=30, columns=["x0", "price"])
    assert all(list(frame.columns) == ["x0", "price"] for frame in stream.chunks())
