import json
from types import SimpleNamespace

import numpy
import pandas
import pytest
import torch

import kalfa
from kalfa.core.errors import BuildError
from kalfa.core.rng import Rng
from kalfa.std.data import splits, stream, table


ROWS = 300
MARKER = 3
SEQUENTIAL = {"use": "sequential", "ratios": [0.7, 0.15, 0.15]}
GIVEN = {"use": "given", "column": "part", "train": 0, "valid": 1, "test": 2}
PARAMS = {"split": SEQUENTIAL, "features": {"x*": "standard_scaler", "marker": None},
          "targets": {"y": "standard_scaler"}, "spectators": ["id"], "drop": ["part"], "batch": 32,
          "stream": {"chunk": 64}}


def stream_frame(rows=ROWS, seed=0):
    rng = numpy.random.default_rng(seed)
    x = rng.normal(size=(rows, 3))
    y = 2.0 * x[:, 0] - x[:, 1] + 0.5 * x[:, 2] + rng.normal(scale=0.1, size=rows)
    order = numpy.arange(rows)
    part = numpy.where(order % 10 < 7, 0, numpy.where(order % 10 < 9, 1, 2))
    return pandas.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "x3": x[:, 2], "marker": order, "y": y, "part": part,
                             "id": order + 1000})


def fake_context(seed=0):
    return SimpleNamespace(seed=seed, rng=Rng(seed), config=SimpleNamespace(directories=[]))


@pytest.fixture
def frame():
    return stream_frame()


@pytest.fixture
def parquet(tmp_path, frame):
    path = tmp_path / "rows.parquet"
    frame.to_parquet(path, index=False, row_group_size=50)
    return path


def fitted(path, ctx=None, **overrides):
    data = table.Table(str(path), **{**PARAMS, **overrides})
    data.fit(fake_context() if ctx is None else ctx)
    return data


def markers(batches):
    return [int(value) for batch in batches for value in batch["x"][:, MARKER].tolist()]


def bounds(data, part):
    return [frame.index.tolist() for frame in data.engine.reader.chunks(["marker"], part.start, part.stop)]


def test_reader_parquet_windows_and_columns(parquet, frame):
    reader = stream.ChunkReader(parquet, chunk=64)
    assert reader.header() == list(frame.columns) and reader.count() == ROWS
    pieces = list(reader.chunks())
    assert all(1 <= len(piece) <= 64 for piece in pieces)
    assert [row for piece in pieces for row in piece.index.tolist()] == list(range(ROWS))
    joined = pandas.concat(pieces)
    numpy.testing.assert_allclose(joined["y"].to_numpy(), frame["y"].to_numpy())
    assert joined["marker"].tolist() == joined.index.tolist()
    window = list(reader.chunks(columns=["marker", "y"], start=120, stop=190))
    assert all(list(piece.columns) == ["marker", "y"] for piece in window)
    assert [row for piece in window for row in piece.index.tolist()] == list(range(120, 190))
    assert [value for piece in window for value in piece["marker"].tolist()] == list(range(120, 190))
    assert [row for piece in reader.chunks(start=295) for row in piece.index.tolist()] == list(range(295, 300))
    assert list(reader.chunks(start=200, stop=200)) == [] and list(reader.chunks(start=400)) == []
    first, second = list(reader.chunks(shard=0, shards=2)), list(reader.chunks(shard=1, shards=2))
    assert [piece.index.tolist() for piece in first] == [piece.index.tolist() for piece in pieces[0::2]]
    assert [piece.index.tolist() for piece in second] == [piece.index.tolist() for piece in pieces[1::2]]


def test_reader_drops_the_pandas_index_column(tmp_path, frame):
    shuffled = frame.iloc[numpy.random.default_rng(0).permutation(ROWS)]
    path = tmp_path / "shuffled.parquet"
    shuffled.to_parquet(path)
    reader = stream.ChunkReader(path, chunk=100)
    assert reader.header() == list(frame.columns)
    pieces = list(reader.chunks())
    assert all(list(piece.columns) == list(frame.columns) for piece in pieces)
    assert [row for piece in pieces for row in piece.index.tolist()] == list(range(ROWS))
    assert pandas.concat(pieces)["marker"].tolist() == shuffled["marker"].tolist()


def test_reader_csv_and_formats(tmp_path, frame):
    path = tmp_path / "rows.csv"
    frame.to_csv(path, index=False)
    reader = stream.ChunkReader(str(path), chunk=64)
    assert reader.header() == list(frame.columns) and reader.count() == ROWS
    pieces = list(reader.chunks())
    assert [len(piece) for piece in pieces] == [64, 64, 64, 64, 44]
    assert [row for piece in pieces for row in piece.index.tolist()] == list(range(ROWS))
    window = list(reader.chunks(columns=["marker"], start=60, stop=70))
    assert [piece.index.tolist() for piece in window] == [list(range(60, 64)), list(range(64, 70))]
    assert [value for piece in window for value in piece["marker"].tolist()] == list(range(60, 70))
    tsv = tmp_path / "rows.tsv"
    frame.to_csv(tsv, index=False, sep="\t")
    assert stream.ChunkReader(tsv, chunk=1000).count() == ROWS
    with pytest.raises(BuildError, match="unknown"):
        stream.ChunkReader(tmp_path / "rows.xlsx")
    with pytest.raises(BuildError, match="positive"):
        stream.ChunkReader(path, chunk=0)
    with pytest.raises(BuildError, match="cannot"):
        stream.ChunkReader(tmp_path / "missing.parquet").count()


def test_sequential_stream_materializes_valid_and_test(parquet, frame):
    data = fitted(parquet)
    assert data.counts == {"train": 210, "valid": 45, "test": 45}
    train = data.datasets["train"]
    assert isinstance(train, stream.StreamDataset) and train.count == 210 and train.rows is None
    assert isinstance(data.datasets["valid"], table.TableDataset)
    assert data.rows("valid").tolist() == list(range(210, 255))
    assert data.rows("test").tolist() == list(range(255, 300))
    with pytest.raises(BuildError, match="the train set is streamed and has no row table"):
        data.rows("train")
    assert data.spectators("train") is None
    assert data.spectators("test")["id"].tolist() == list(range(1255, 1300))
    assert data.feature_names == ["x1", "x2", "x3", "marker"] and data.target_names == ["y"]
    test = data.datasets["test"]
    numpy.testing.assert_allclose(data.inverse("y", test.targets["y"]).numpy().reshape(-1),
                                  frame.loc[255:, "y"].to_numpy(), rtol=1e-4)
    assert test.x[:, MARKER].tolist() == list(range(255, 300))
    assert data.report()["rows"] == {"train": 210, "valid": 45, "test": 45}


def test_train_loader_yields_every_row_once_per_epoch(parquet):
    data = fitted(parquet)
    loaders = data.loaders()
    assert set(loaders) == {"train", "valid", "test"}
    loader = loaders["train"]
    first = list(loader)
    assert set(first[0]) == {"x", "y"} and tuple(first[0]["x"].shape) == (32, 4)
    assert first[0]["x"].dtype == torch.float32 and tuple(first[0]["y"].shape) == (32, 1)
    epochs = [markers(first), markers(loader), markers(loader)]
    for epoch in epochs:
        assert sorted(epoch) == list(range(210))
    assert epochs[0] != list(range(210)) and epochs[0] != epochs[1]
    position = 0
    for rows in bounds(data, data.datasets["train"].part):
        assert sorted(epochs[0][position:position + len(rows)]) == rows
        position += len(rows)
    assert position == 210
    again = fitted(parquet)
    assert markers(again.loaders()["train"]) == epochs[0]
    valid = list(loaders["valid"])
    assert markers(valid) == list(range(210, 255))
    plain = fitted(parquet, ctx=SimpleNamespace(seed=0, rng=None, config=None))
    assert markers(plain.loaders()["train"]) == list(range(210))
    still = fitted(parquet, batch={"size": 32, "shuffle": False})
    assert markers(still.loaders()["train"]) == list(range(210))


def test_fit_sample_respects_fit_rows(parquet, frame):
    data = fitted(parquet, stream={"chunk": 64, "fit_rows": 50})
    sets = data.engine.sets_of(fake_context())
    sample = data.engine.fit_sample(sets)
    assert sample.index.tolist() == list(range(50)) and "part" not in sample.columns
    scaler = data.blocks[0].chain[0]
    numpy.testing.assert_allclose(scaler.mean_, frame.loc[:49, ["x1", "x2", "x3"]].mean().to_numpy(), rtol=1e-6)
    assert data.counts["train"] == 210
    whole = fitted(parquet, stream={"chunk": 64, "fit_rows": 1000})
    assert len(whole.engine.fit_sample(whole.engine.sets_of(fake_context()))) == 210
    numpy.testing.assert_allclose(whole.blocks[0].chain[0].mean_,
                                  frame.loc[:209, ["x1", "x2", "x3"]].mean().to_numpy(), rtol=1e-6)


def test_given_split_filters_by_the_column(parquet, frame):
    data = fitted(parquet, split=GIVEN)
    wanted = {name: frame.index[frame["part"] == value].tolist() for name, value in (("train", 0), ("valid", 1),
                                                                                       ("test", 2))}
    assert data.counts == {"train": 210, "valid": 60, "test": 30}
    assert sorted(markers(data.loaders()["train"])) == wanted["train"]
    assert data.rows("valid").tolist() == wanted["valid"] and data.rows("test").tolist() == wanted["test"]
    assert data.feature_names == ["x1", "x2", "x3", "marker"] and "part" not in data.spectators("test").columns
    part = data.datasets["train"].part
    assert (part.start, part.stop, part.where) == (0, ROWS, 0)
    custom = fitted(parquet, split=splits.GivenSplit("part", train=0, valid=None, test=1))
    assert custom.counts == {"train": 210, "test": 60}
    with pytest.raises(BuildError, match="not in the table"):
        fitted(parquet, split={"use": "given", "column": "nope"})
    with pytest.raises(BuildError, match="no train rows"):
        fitted(parquet, split={"use": "given", "column": "part", "train": 7})


def test_other_splits_raise(parquet):
    for split in ([0.8, 0.2], {"use": "kfold", "folds": 3, "fold": 0}, {"use": "random", "ratios": [0.8, 0.2]}):
        with pytest.raises(BuildError, match="a stream needs a sequential or given split"):
            fitted(parquet, split=split)
    with pytest.raises(BuildError, match="no group"):
        fitted(parquet, split={"use": "sequential", "ratios": [0.8, 0.2], "group": "part"})
    with pytest.raises(BuildError, match="no feed"):
        fitted(parquet, feed=SimpleNamespace(group=None))


def test_filters_apply_per_set(parquet, frame):
    spec = {"all": "x1 > -1.0", "train": "x2 > 0", "test": "y > 0"}
    data = fitted(parquet, filter=spec)
    kept = frame[frame["x1"] > -1.0]
    train = kept.loc[:209]
    train = train[train["x2"] > 0].index.tolist()
    valid = kept.loc[210:254].index.tolist()
    test = kept.loc[255:]
    test = test[test["y"] > 0].index.tolist()
    assert data.counts == {"train": len(train), "valid": len(valid), "test": len(test)}
    assert data.datasets["train"].count == len(train) < 210
    assert sorted(markers(data.loaders()["train"])) == train
    assert data.rows("valid").tolist() == valid and data.rows("test").tolist() == test
    emptied = fitted(parquet, filter={"test": "y > 1e9"})
    assert set(emptied.counts) == {"train", "valid"}
    short = fitted(parquet, split={"use": "sequential", "ratios": [0.85, 0.15]})
    assert short.counts == {"train": 255, "valid": 45}
    with pytest.raises(BuildError, match="no train rows"):
        fitted(parquet, filter="x1 > 1e9")
    with pytest.raises(BuildError, match="query"):
        fitted(parquet, filter="nope > 1")


def test_workers_split_the_chunks(parquet, monkeypatch):
    data = fitted(parquet, batch={"size": 32, "shuffle": False})
    dataset = data.datasets["train"]
    chunks = bounds(data, dataset.part)
    assert len(chunks) > 2
    seen = []
    for shard in (0, 1):
        monkeypatch.setattr(stream, "get_worker_info", lambda shard=shard: SimpleNamespace(id=shard, num_workers=2))
        rows = [int(item["x"][MARKER]) for item in dataset]
        assert rows == [row for rows in chunks[shard::2] for row in rows]
        seen.append(set(rows))
    assert seen[0].isdisjoint(seen[1]) and seen[0] | seen[1] == set(range(210))


def test_loader_for_a_frame_still_works(parquet, frame):
    data = fitted(parquet)
    fresh = stream_frame(rows=12, seed=4)
    fresh.index = pandas.RangeIndex(700, 712)
    loader = data.loader_for(fresh)
    batch = next(iter(loader))
    assert set(batch) == {"x", "y"} and tuple(batch["x"].shape) == (12, 4)
    assert loader.dataset.rows.tolist() == list(range(700, 712))
    expected = data.blocks[0].chain[0].transform(fresh[["x1", "x2", "x3"]].to_numpy()).astype("float32")
    numpy.testing.assert_allclose(batch["x"][:, :3].numpy(), expected, rtol=1e-5)


def test_csv_stream_matches_parquet(parquet, frame, tmp_path):
    path = tmp_path / "rows.csv"
    frame.to_csv(path, index=False)
    data = fitted(path, filter={"valid": "x3 > 0"})
    reference = fitted(parquet, filter={"valid": "x3 > 0"})
    assert data.counts == reference.counts
    assert data.rows("valid").tolist() == reference.rows("valid").tolist()
    assert sorted(markers(data.loaders()["train"])) == list(range(210))


def test_save_load_and_rebuild(parquet, tmp_path):
    data = fitted(parquet)
    record = tmp_path / "record"
    data.save(record)
    fields = json.loads((record / "fitted" / "fields.json").read_text())
    assert fields["drops"] == ["part"] and fields["targets"] == ["y"]
    loaded = table.Table(str(parquet), **PARAMS).load(record)
    loaded.rebuild(fake_context())
    assert loaded.counts == data.counts and loaded.rows("test").tolist() == data.rows("test").tolist()
    assert markers(loaded.loaders()["train"]) == markers(fitted(parquet).loaders()["train"])


def test_options_are_checked(parquet):
    assert table.Table(str(parquet), split=SEQUENTIAL, stream=True).engine.options == stream.DEFAULTS
    with pytest.raises(BuildError, match="unknown keys"):
        table.Table(str(parquet), split=SEQUENTIAL, stream={"chunks": 10})
    with pytest.raises(BuildError, match="positive integer"):
        table.Table(str(parquet), split=SEQUENTIAL, stream={"chunk": 0})
    with pytest.raises(BuildError, match="positive integer"):
        table.Table(str(parquet), split=SEQUENTIAL, stream={"fit_rows": True})


def test_run_streams_a_table_end_to_end(parquet, frame, tmp_path):
    record = tmp_path / "record"
    config = {
        "seed": 1,
        "device": "cpu",
        "data": {"use": "table", "file": str(parquet), "split": SEQUENTIAL, "features": {"x*": "standard_scaler"},
                 "targets": {"y": "standard_scaler"}, "spectators": ["id"], "drop": ["marker", "part"],
                 "batch": 32, "stream": {"chunk": 64}},
        "models": {"net": {"use": "mlp", "hidden": [8], "out": 1}},
        "losses": {"mse": "mse"},
        "optimizers": {"main": {"use": "adam", "lr": 0.01, "over": "@models.net", "minimizes": "@losses.mse"}},
        "train": {"epochs": 2},
        "record": str(record),
    }
    result = kalfa.run(config)
    assert result.context.turn == 2
    history = [json.loads(line) for line in (record / "history.jsonl").read_text().splitlines()]
    assert [line["turn"] for line in history] == [1, 2]
    assert {"train/mse", "val/mse", "test/mse"} <= set(history[-1])
    assert history[-1]["update"] == 2 * 7
    predictions = pandas.read_parquet(record / "predictions.parquet")
    assert predictions["row"].tolist() == list(range(255, 300))
    assert predictions["id"].tolist() == list(range(1255, 1300))
    numpy.testing.assert_allclose(predictions["y"].to_numpy(), frame.loc[255:, "y"].to_numpy(), rtol=1e-4)
    assert numpy.isfinite(predictions["pred_net"].to_numpy()).all()
    again = kalfa.predict(record)
    assert again.table["row"].tolist() == list(range(255, 300))
    numpy.testing.assert_allclose(again.table["y"].to_numpy(), predictions["y"].to_numpy())
