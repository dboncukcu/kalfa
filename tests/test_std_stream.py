"""The lazy set: chunked sources, filters and windows on the stream, streamed fit, the stream dataset and loader."""

import numpy
import pandas
import pytest
import torch

import kalfa  # noqa: F401
from kalfa.std.data import filter as filter_rows
from kalfa.std.data import filter_set
from kalfa.std.feed import StreamDataset, dataset_size, sized, table, window
from kalfa.std.loader import torch as torch_loader
from kalfa.std.pre import apply, fit, one_hot, standard_scaler
from kalfa.std.source import csv_stream, header, parquet_stream
from kalfa.std.split import given, kfold, random as random_split, sequential
from kalfa.std.stream import positions
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
    assert header("/source/kalfa/parquet_stream", {"path": str(directory / "housing.parquet")})["rows"] == 100
    csv = csv_stream(str(directory / "housing.csv"), chunk=40)
    assert positions(csv).tolist() == list(range(100)) and csv.columns == list(frame.columns)
    assert header("/source/kalfa/csv_stream", {"path": str(directory / "housing.csv")})["rows"] == 100


def test_filters_and_windows_apply_on_the_stream(housing):
    directory, frame = housing
    stream = parquet_stream(str(directory / "housing.parquet"), chunk=30)
    odd = filter_rows(stream, "kind == 'odd'")
    assert positions(odd).tolist() == list(range(0, 100, 4)) and odd.count() == 25
    parts = sequential(stream, [0.7, 0.2, 0.1])
    assert [part.rows for part in parts.values()] == [70, 20, 10]
    assert positions(parts["valid"]).tolist() == list(range(70, 90))
    assert positions(filter_set(parts["train"], "train", [{"query": "price > 200", "sets": ["train"]}])).tolist() == \
        [position for position in range(70) if frame["price"][position] > 200]
    assert positions(filter_set(parts["test"], "test", [{"query": "price > 1e9", "sets": ["train"]}])).tolist() == \
        list(range(90, 100))
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
    pres = {"scale": standard_scaler(), "hot": one_hot()}
    lazy = fit(stream, fields, pres, [])
    eager = fit(frame, fields, {"scale": standard_scaler(), "hot": one_hot()}, [])
    assert lazy.features == eager.features and lazy.dtypes == eager.dtypes
    for name in ("x0", "x1", "x2"):
        assert lazy.fitted["scale"][name].scaler.mean_ == pytest.approx(eager.fitted["scale"][name].scaler.mean_)
    assert lazy.fitted["hot"]["kind"].columns("kind") == eager.fitted["hot"]["kind"].columns("kind")
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
    prep = fit(stream, fields, {"scale": standard_scaler()}, ["kind"])
    parts = sequential(stream, [0.7, 0.2, 0.1])
    train = table(apply(parts["train"], prep, "train"))
    assert isinstance(train, StreamDataset) and sized(train) is None and dataset_size(train) == 70
    loader = torch_loader(train, "train", {"size": 16, "buffer": 8})
    torch.manual_seed(3)
    first = [batch["x"] for batch in loader]
    order_first = train.rows().tolist()
    torch.manual_seed(3)
    list(loader)
    assert train.rows().tolist() == order_first and sorted(order_first) == list(range(70))
    assert order_first != list(range(70)) and first[0].shape == (16, 3)
    test = torch_loader(table(apply(parts["test"], prep, "test")), "test", {"size": 16})
    batches = list(test)
    assert test.dataset.rows().tolist() == list(range(90, 100)) and batches[0]["price"].shape == (10,)
    with pytest.raises(ValueError, match="cannot be counted"):
        torch_loader(train, "train", {"size": 16, "balanced": True})
    with pytest.raises(ValueError, match="workers: 0"):
        torch_loader(train, "train", {"size": 16, "workers": 2})
    with pytest.raises(ValueError, match="cannot be counted"):
        train.labels("price")
    with pytest.raises(ValueError, match="table in memory"):
        window(apply(parts["train"], prep, "train"), None, size=2, horizon=1)
    empty = table(apply(filter_rows(parts["train"], "price > 1e9"), prep, "train"))
    with pytest.raises(ValueError, match="yields no rows"):
        list(torch_loader(empty, "train", {"size": 4}))
    assert list(torch_loader(table(apply(parts["train"].empty(), prep, "valid")), "valid", {"size": 4})) == []
