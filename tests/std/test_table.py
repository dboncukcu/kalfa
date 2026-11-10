import json
import logging

import numpy
import pandas
import pytest
import torch
from torch.utils.data import RandomSampler, SequentialSampler

from kalfa.core import aliases
from kalfa.core.errors import BuildError
from kalfa.std.data import pre, splits, table


class FakeRng:
    def stream(self, name, device=None):
        return torch.Generator().manual_seed(11)


class FakeContext:
    def __init__(self, seed=7, rng=True):
        self.seed = seed
        self.rng = FakeRng() if rng else None


def housing_frame(rows=200, seed=0):
    rng = numpy.random.default_rng(seed)
    x = rng.normal(size=(rows, 4))
    noise = rng.normal(scale=0.1, size=rows)
    price = numpy.exp(0.5 * x[:, 0] - 0.3 * x[:, 1] + noise) * 100.0 + 50.0
    frame = pandas.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "x3": x[:, 2], "x4": x[:, 3], "price": price})
    frame["city"] = rng.choice(["ankara", "bursa", "izmir"], size=rows)
    frame["id"] = numpy.arange(1000, 1000 + rows)
    return frame


def blobs_frame(rows=200, seed=1):
    rng = numpy.random.default_rng(seed)
    label = rng.choice([0, 1, 2], size=rows, p=[0.6, 0.3, 0.1])
    centers = numpy.array([[0.0, 0.0], [3.0, 3.0], [-3.0, 3.0]])
    points = centers[label] + rng.normal(scale=0.5, size=(rows, 2))
    return pandas.DataFrame({"f1": points[:, 0], "f2": points[:, 1], "label": label})


@pytest.fixture
def housing(tmp_path):
    frame = housing_frame()
    path = tmp_path / "housing.parquet"
    frame.to_parquet(path)
    return path, frame


@pytest.fixture
def blobs(tmp_path):
    frame = blobs_frame()
    path = tmp_path / "blobs.parquet"
    frame.to_parquet(path)
    return path, frame


HOUSING_PARAMS = {"split": [0.7, 0.15, 0.15], "features": {"x*": "standard_scaler"},
                  "targets": {"price": "standard_scaler"}, "spectators": ["id"], "drop": ["city"], "batch": 32}


def fitted_housing(path, ctx=None, **overrides):
    params = {**HOUSING_PARAMS, **overrides}
    data = table.Table(str(path), **params)
    data.fit(FakeContext() if ctx is None else ctx)
    return data


def test_split_sizes_and_row_identity(housing):
    path, frame = housing
    data = fitted_housing(path)
    assert data.counts == {"train": 140, "valid": 30, "test": 30}
    assert data.split.seed == 7
    rows = {name: set(data.rows(name).tolist()) for name in data.counts}
    assert rows["train"].isdisjoint(rows["valid"]) and rows["train"].isdisjoint(rows["test"])
    assert rows["valid"].isdisjoint(rows["test"])
    assert rows["train"] | rows["valid"] | rows["test"] == set(range(200))
    again = fitted_housing(path)
    assert again.rows("test").tolist() == data.rows("test").tolist()
    other_seed = fitted_housing(path, ctx=FakeContext(seed=8))
    assert other_seed.rows("test").tolist() != data.rows("test").tolist()


def test_feature_tensor_dtype_and_width(housing):
    path, frame = housing
    data = fitted_housing(path)
    train = data.datasets["train"]
    assert train.x.dtype == torch.float32 and tuple(train.x.shape) == (140, 4)
    assert data.feature_names == ["x1", "x2", "x3", "x4"] and data.n_features == 4
    assert abs(float(train.x.mean())) < 1e-5 and abs(float(train.x.square().mean()) - 1.0) < 1e-4
    price = train.targets["price"]
    assert price.dtype == torch.float32 and tuple(price.shape) == (140, 1)
    item = train[0]
    assert set(item) == {"x", "price"} and tuple(item["x"].shape) == (4,) and tuple(item["price"].shape) == (1,)
    assert data.target_names == ["price"] and data.n_classes is None and data.classes is None
    assert data.class_weights is None and data.inputs == ("x",)
    assert len(train) == 140 and train.rows.tolist() == data.rows("train").tolist()


def test_inverse_round_trip_standard_scaler(housing):
    path, frame = housing
    data = fitted_housing(path)
    encoded = data.datasets["test"].targets["price"]
    raw = frame.loc[data.rows("test"), "price"].to_numpy()
    back = data.inverse("price", encoded)
    assert isinstance(back, torch.Tensor) and back.dtype == torch.float32 and tuple(back.shape) == (30, 1)
    numpy.testing.assert_allclose(back.numpy().reshape(-1), raw, rtol=1e-4)
    back_numpy = data.inverse("price", encoded.numpy().reshape(-1))
    assert isinstance(back_numpy, numpy.ndarray) and back_numpy.dtype == numpy.float32
    numpy.testing.assert_allclose(back_numpy, raw, rtol=1e-4)
    assert data.rescales("price") is True and data.rescales("nothing") is False
    with pytest.raises(BuildError, match="unknown target"):
        data.inverse("nothing", encoded)


@pytest.mark.parametrize("spec", [
    [pre.Log(base=2.0, norm=10.0), "standard_scaler"],
    [{"use": "log", "base": 2.0, "norm": 10.0}, "standard_scaler"],
    ["abs", "log", "standard_scaler"],
])
def test_log_chain_round_trip(housing, spec):
    path, frame = housing
    data = fitted_housing(path, targets={"price": spec})
    target = data.fitted_target("price")
    assert table.step_names(target.chain)[-2:] == ["Log", "StandardScaler"]
    raw = frame.loc[data.rows("valid"), "price"].to_numpy()
    encoded = data.datasets["valid"].targets["price"]
    numpy.testing.assert_allclose(data.inverse("price", encoded).numpy().reshape(-1), raw, rtol=1e-4)
    assert data.rescales("price") is True


def test_one_hot_expands_columns(housing):
    path, frame = housing
    data = fitted_housing(path, features={"x*": "standard_scaler", "city": "one_hot"}, drop=None)
    assert data.n_features == 7
    assert data.feature_names == ["x1", "x2", "x3", "x4", "city_ankara", "city_bursa", "city_izmir"]
    x = data.datasets["train"].x
    assert tuple(x.shape) == (140, 7)
    assert torch.equal(x[:, 4:].sum(dim=1), torch.ones(140))
    assert data.report()["columns"]["city_bursa"] == ["OneHotEncoder"]
    unknown = housing_frame(rows=3, seed=2).assign(city=["ankara", "paris", "izmir"])
    batch = next(iter(data.loader_for(unknown)))
    assert batch["x"][:, 4:].sum(dim=1).tolist() == [1.0, 0.0, 1.0]


def test_label_target_makes_classification(blobs):
    path, frame = blobs
    data = table.Table(str(path), split=[0.8, 0.2], features={"f*": "standard_scaler"}, targets={"label": "label"},
                       batch=16)
    data.fit(FakeContext())
    assert data.counts == {"train": 160, "valid": 40}
    y = data.datasets["train"].targets["label"]
    assert y.dtype == torch.int64 and tuple(y.shape) == (160,)
    assert data.n_classes == 3 and data.classes == [0, 1, 2]
    weights = data.class_weights
    assert weights.dtype == torch.float32 and tuple(weights.shape) == (3,)
    assert abs(float(weights.mean()) - 1.0) < 1e-6
    counts = torch.bincount(y, minlength=3).float()
    expected = 1.0 / counts
    assert torch.allclose(weights, expected / expected.mean())
    assert int(weights.argmax()) == int(counts.argmin())
    assert data.rescales("label") is False
    assert data.decode("label", torch.tensor([2, 0, 1])).tolist() == [2, 0, 1]
    assert data.decode("label", torch.tensor([[0.1, 0.9, 0.0], [2.0, 0.0, 0.0]])).tolist() == [1, 0]
    assert data.decode("label", numpy.array([[2], [1]])).tolist() == [2, 1]
    with pytest.raises(BuildError, match="width 3"):
        data.decode("label", numpy.zeros((4, 2)))
    same = data.inverse("label", y[:5])
    assert same.dtype == torch.float32 and same.tolist() == [float(value) for value in y[:5].tolist()]
    assert data.report()["classes"] == [0, 1, 2]
    batch = next(iter(data.loaders()["train"]))
    assert set(batch) == {"x", "label"} and tuple(batch["label"].shape) == (16,)
    with pytest.raises(BuildError, match="label"):
        data.decode("nothing", y)


def test_string_label_target_and_fitted_files(blobs, tmp_path):
    path, frame = blobs
    frame = frame.assign(kind=numpy.array(["low", "mid", "high"])[frame["label"].to_numpy()])
    frame.to_parquet(path)
    params = {"split": [0.75, 0.25], "features": {"f*": "standard_scaler"}, "targets": {"kind": "label"},
              "drop": ["label"], "batch": 16}
    data = table.Table(str(path), **params)
    data.fit(FakeContext())
    assert data.classes == ["high", "low", "mid"] and data.n_classes == 3
    y = data.datasets["valid"].targets["kind"]
    raw = frame.loc[data.rows("valid"), "kind"].to_numpy()
    assert data.decode("kind", y).tolist() == raw.tolist()
    data.save(tmp_path / "record")
    fields = json.loads((tmp_path / "record" / "fitted" / "fields.json").read_text())
    assert fields["classes"] == ["high", "low", "mid"] and fields["targets"] == ["kind"]
    loaded = table.Table("gone.parquet", **params).load(tmp_path / "record")
    assert loaded.classes == data.classes and torch.equal(loaded.class_weights, data.class_weights)
    batch = next(iter(loaded.loader_for(frame.head(5))))
    assert batch["kind"].dtype == torch.int64
    assert loaded.decode("kind", batch["kind"]).tolist() == frame["kind"].head(5).tolist()


def test_loaders_batches(housing):
    path, frame = housing
    data = fitted_housing(path, batch={"size": 16, "eval_size": 64, "drop_last": True, "shuffle": True})
    loaders = data.loaders()
    assert set(loaders) == {"train", "valid", "test"}
    batch = next(iter(loaders["train"]))
    assert set(batch) == {"x", "price"}
    assert tuple(batch["x"].shape) == (16, 4) and batch["x"].dtype == torch.float32
    assert tuple(batch["price"].shape) == (16, 1) and batch["price"].dtype == torch.float32
    assert len(loaders["train"]) == 140 // 16
    assert loaders["train"].batch_size == 16 and loaders["valid"].batch_size == 64
    assert isinstance(loaders["train"].sampler, RandomSampler) and loaders["train"].sampler.generator is not None
    assert isinstance(loaders["valid"].sampler, SequentialSampler)
    assert len(loaders["valid"]) == 1 and len(loaders["test"]) == 1
    valid = next(iter(loaders["valid"]))
    assert tuple(valid["x"].shape) == (30, 4)
    numpy.testing.assert_allclose(valid["x"].numpy(), data.datasets["valid"].x.numpy())
    again = fitted_housing(path, batch={"size": 16, "eval_size": 64, "drop_last": True, "shuffle": True})
    assert torch.equal(next(iter(again.loaders()["train"]))["x"], batch["x"])
    plain = fitted_housing(path, ctx=FakeContext(rng=False))
    train = plain.loaders()["train"]
    assert train.batch_size == 32 and train.sampler.generator is None and plain.loaders()["test"].batch_size == 32
    assert len(train) == -(-140 // 32)
    with pytest.raises(BuildError, match="unknown keys"):
        table.Table(str(path), split=[0.8, 0.2], batch={"size": 8, "pin": True})
    with pytest.raises(BuildError, match="positive integer"):
        table.Table(str(path), split=[0.8, 0.2], batch=0)


def test_loader_for_new_rows(housing):
    path, frame = housing
    data = fitted_housing(path)
    fresh = housing_frame(rows=10, seed=5)
    fresh.index = pandas.RangeIndex(500, 510)
    loader = data.loader_for(fresh)
    batch = next(iter(loader))
    assert set(batch) == {"x", "price"} and tuple(batch["x"].shape) == (10, 4)
    assert loader.dataset.rows.tolist() == list(range(500, 510))
    assert isinstance(loader.sampler, SequentialSampler)
    scaler = data.blocks[0].chain[0]
    expected = scaler.transform(fresh[["x1", "x2", "x3", "x4"]].to_numpy()).astype("float32")
    numpy.testing.assert_allclose(batch["x"].numpy(), expected, rtol=1e-5)
    unlabeled = fresh.drop(columns=["price"])
    assert set(next(iter(data.loader_for(unlabeled)))) == {"x"}
    with pytest.raises(BuildError, match="feature columns"):
        data.loader_for(fresh.drop(columns=["x2"]))
    with pytest.raises(BuildError, match="DataFrame"):
        data.loader_for([1, 2, 3])


def test_save_and_load_round_trip(housing, tmp_path):
    path, frame = housing
    features = {"x*": "standard_scaler", "city": "one_hot"}
    data = fitted_housing(path, features=features, drop=None)
    record = tmp_path / "run"
    data.save(record)
    assert (record / "fitted" / "preprocessors.joblib").exists()
    fields = json.loads((record / "fitted" / "fields.json").read_text())
    assert fields["feature_names"] == data.feature_names and fields["targets"] == ["price"]
    assert fields["features"][1]["chain"] == ["OneHotEncoder"] and fields["drops"] == []
    loaded = table.Table("missing.parquet", **{**HOUSING_PARAMS, "features": features, "drop": None}).load(record)
    assert loaded.feature_names == data.feature_names and loaded.n_features == 7 and loaded.classes is None
    fresh = housing_frame(rows=12, seed=9)
    want = next(iter(data.loader_for(fresh)))
    got = next(iter(loaded.loader_for(fresh)))
    assert torch.equal(want["x"], got["x"]) and torch.equal(want["price"], got["price"])
    numpy.testing.assert_allclose(loaded.inverse("price", got["price"]).numpy(),
                                  data.inverse("price", want["price"]).numpy())
    assert loaded.rescales("price") is True and loaded.report()["rows"] == {}
    with pytest.raises(BuildError, match="not fitted"):
        table.Table("missing.parquet", split=[0.8, 0.2]).save(record)
    with pytest.raises(BuildError, match="fields.json"):
        table.Table("missing.parquet", split=[0.8, 0.2]).load(tmp_path / "empty")


def test_sequential_split():
    frame = pandas.DataFrame({"v": numpy.arange(100), "g": ["a"] * 60 + ["b"] * 40})
    sets = splits.SequentialSplit([0.5, 0.25, 0.25]).split(frame)
    assert sets["train"].index.tolist() == list(range(50))
    assert sets["valid"].index.tolist() == list(range(50, 75))
    assert sets["test"].index.tolist() == list(range(75, 100))
    grouped = splits.SequentialSplit([0.5, 0.5], group="g").split(frame)
    assert set(grouped) == {"train", "valid"}
    assert grouped["train"].index.tolist() == list(range(30)) + list(range(60, 80))
    assert grouped["valid"].index.tolist() == list(range(30, 60)) + list(range(80, 100))
    with pytest.raises(BuildError, match="group column"):
        splits.SequentialSplit([0.5, 0.5], group="nope").split(frame)


def test_kfold_split():
    frame = pandas.DataFrame({"v": numpy.arange(200)})
    held = []
    for fold in range(5):
        sets = splits.KFoldSplit(5, fold, seed=3).split(frame)
        assert set(sets) == {"train", "test"}
        assert len(sets["test"]) == 40 and len(sets["train"]) == 160
        assert set(sets["train"].index) | set(sets["test"].index) == set(range(200))
        held.append(set(sets["test"].index))
    assert all(first.isdisjoint(second) for position, first in enumerate(held) for second in held[position + 1:])
    assert set().union(*held) == set(range(200))
    carved = splits.KFoldSplit(5, 2, seed=3, valid_ratio=0.25).split(frame)
    assert {name: len(part) for name, part in carved.items()} == {"train": 120, "valid": 40, "test": 40}
    assert carved["test"].index.tolist() == sorted(held[2])
    for folds, fold in ((1, 0), (5, 5), (5, -1), (5.0, 1)):
        with pytest.raises(BuildError):
            splits.KFoldSplit(folds, fold)


def test_given_split():
    frame = pandas.DataFrame({"v": numpy.arange(10), "part": ["train"] * 6 + ["valid"] * 2 + ["test"] * 2})
    sets = splits.GivenSplit("part").split(frame)
    assert {name: part.index.tolist() for name, part in sets.items()} == {"train": [0, 1, 2, 3, 4, 5],
                                                                          "valid": [6, 7], "test": [8, 9]}
    custom = splits.GivenSplit("part", valid=None, test="valid").split(frame)
    assert set(custom) == {"train", "test"} and custom["test"].index.tolist() == [6, 7]
    with pytest.raises(BuildError, match="not in the table"):
        splits.GivenSplit("nope").split(frame)
    with pytest.raises(BuildError, match="holds none"):
        splits.GivenSplit("part", train="a", valid="b", test="c").split(frame)


def test_random_split_ratios():
    frame = pandas.DataFrame({"v": numpy.arange(10)})
    assert set(splits.RandomSplit([0.8, 0.2], seed=0).split(frame)) == {"train", "valid"}
    assert set(splits.RandomSplit([0.8, 0.0, 0.2], seed=0).split(frame)) == {"train", "test"}
    assert set(splits.RandomSplit([1.0], seed=0).split(frame)) == {"train"}
    first = splits.RandomSplit([0.5, 0.5], seed=4).split(frame)["train"].index.tolist()
    assert first == splits.RandomSplit([0.5, 0.5], seed=4).split(frame)["train"].index.tolist()
    assert first == sorted(first) and first != list(range(5))
    for bad in ([0.5, 0.6], [-0.1, 1.1], [0.2, 0.2, 0.2, 0.4], "abc", []):
        with pytest.raises(BuildError):
            splits.RandomSplit(bad)


def test_built_split_object_takes_the_context_seed(housing):
    path, frame = housing
    data = fitted_housing(path, split=splits.KFoldSplit(4, 1))
    assert data.split.seed == 7 and data.counts == {"train": 150, "test": 50}
    assert set(data.loaders()) == {"train", "test"}
    other = fitted_housing(path, split=splits.KFoldSplit(4, 1))
    assert data.rows("test").tolist() == other.rows("test").tolist()
    given = frame.assign(part=numpy.where(frame.index < 150, "train", "test"))
    given.to_parquet(path)
    by_column = fitted_housing(path, split=splits.GivenSplit("part"), drop=["city", "part"])
    assert by_column.counts == {"train": 150, "test": 50} and by_column.rows("test").tolist() == list(range(150, 200))


def test_filter_per_set(housing):
    path, frame = housing
    median = float(frame["price"].median())
    data = fitted_housing(path, filter={"all": "x1 > -1.5", "test": f"price > {median}"})
    kept = frame[frame["x1"] > -1.5]
    assert len(kept) < 200 and data.counts["train"] == int(len(kept) * 0.7 + 1e-9)
    assert data.counts["test"] < int(len(kept) * 0.15 + 1e-9) + 1
    assert set(data.rows("test")) <= set(kept.index)
    assert (frame.loc[data.rows("test"), "price"] > median).all()
    assert (frame.loc[data.rows("train"), "x1"] > -1.5).all()
    assert not (frame.loc[data.rows("train"), "price"] > median).all()
    fresh = housing_frame(rows=20, seed=3)
    fresh.loc[fresh.index[0], "x1"] = -3.0
    loader = data.loader_for(fresh)
    assert len(loader.dataset) == int((fresh["x1"] > -1.5).sum()) < 20
    assert loader.dataset.rows.tolist() == fresh.index[fresh["x1"] > -1.5].tolist()
    plain = fitted_housing(path, filter="x1 > -1.5")
    assert sum(plain.counts.values()) == len(kept)
    with pytest.raises(BuildError, match="unknown keys"):
        table.Table(str(path), split=[0.8, 0.2], filter={"nope": "x1 > 0"})
    with pytest.raises(BuildError, match="no train rows"):
        fitted_housing(path, filter="price > 1e9")
    with pytest.raises(BuildError, match="leaves the test set empty"):
        fitted_housing(path, filter={"test": "price > 1e9"})
    with pytest.raises(BuildError, match="query"):
        fitted_housing(path, filter="nope > 1")


def test_unmatched_columns_warn_and_drop(housing, caplog):
    path, frame = housing
    with caplog.at_level(logging.WARNING, logger="kalfa.data"):
        data = fitted_housing(path, drop=None)
    assert data.feature_names == ["x1", "x2", "x3", "x4"]
    assert any("city" in record.message and "dropped" in record.message for record in caplog.records)
    assert list(data.spectator_frames["test"].columns) == ["id"]
    assert data.spectator_frames["test"].index.tolist() == data.rows("test").tolist()
    assert data.spectator_frames["test"]["id"].tolist() == [1000 + row for row in data.rows("test").tolist()]
    assert data.spectator_patterns == ["id"]
    assert data.spectators("test") is data.spectator_frames["test"] and data.spectators("nothing") is None


def test_specific_pattern_wins(housing):
    path, frame = housing
    data = fitted_housing(path, features={"x*": "standard_scaler", "x1": "minmax_scaler"})
    assert data.feature_names == ["x2", "x3", "x4", "x1"]
    assert data.report()["columns"] == {"x2": ["StandardScaler"], "x3": ["StandardScaler"],
                                        "x4": ["StandardScaler"], "x1": ["MinMaxScaler"]}
    x1 = data.datasets["train"].x[:, 3]
    assert abs(float(x1.min())) < 1e-6 and abs(float(x1.max()) - 1.0) < 1e-6
    with pytest.raises(BuildError, match="equal specificity"):
        fitted_housing(path, features={"x?": "standard_scaler", "?1": "minmax_scaler"})
    with pytest.raises(BuildError, match="matches no column"):
        fitted_housing(path, features={"x*": "standard_scaler", "zzz*": "standard_scaler"})
    with pytest.raises(BuildError, match="not numeric"):
        fitted_housing(path, features={"x*": "standard_scaler", "city": None}, drop=None)
    with pytest.raises(BuildError, match="target columns"):
        fitted_housing(path, targets={"cost": None})


def test_feed_stream_and_split_are_checked(housing):
    path, _ = housing
    assert table.Table(str(path), split=[0.8, 0.2], feed=None).feed is None
    assert table.Table(str(path), split=[0.8, 0.2], stream=True).engine is not None
    with pytest.raises(BuildError, match="split is required"):
        table.Table(str(path))
    with pytest.raises(BuildError, match="named x"):
        table.Table(str(path), split=[0.8, 0.2], targets={"x": None})
    with pytest.raises(BuildError, match="not found"):
        table.Table("nowhere.parquet", split=[0.8, 0.2]).fit(FakeContext())
    with pytest.raises(BuildError, match="not fitted"):
        table.Table(str(path), split=[0.8, 0.2]).loaders()


def test_check_flags_a_stream_with_a_random_split_and_bad_ratios(housing):
    path, _ = housing
    base = {"file": str(path), "features": {"x*": "standard_scaler"}, "targets": {"price": None}}

    def problems_of(**params):
        problems = []
        table.Table.check({**base, **params}, problems)
        return [(problem.code, problem.path, problem.message) for problem in problems]

    found = problems_of(split=[0.8, 0.2], stream=True)
    assert [(code, where) for code, where, _ in found] == [("data_value", "data.split")]
    assert "a stream needs a sequential or given split" in found[0][2]
    folded = problems_of(split={"use": "kfold", "folds": 3, "fold": 0}, stream={"chunk": 64})
    assert [(code, where) for code, where, _ in folded] == [("data_value", "data.split")]
    assert problems_of(split={"use": "sequential", "ratios": [0.8, 0.2]}, stream=True) == []
    assert problems_of(split={"use": "given", "column": "part"}, stream=True) == []
    assert problems_of(split=[0.8, 0.2]) == []
    bad = problems_of(split=[0.5, 0.5, 0.5])
    assert [(code, where) for code, where, _ in bad] == [("data_value", "data.split")]
    assert "sum to 1" in bad[0][2]


def test_reads_csv_and_reports(housing, tmp_path):
    path, frame = housing
    csv = tmp_path / "housing.csv"
    frame.to_csv(csv, index=False)
    data = fitted_housing(csv)
    assert data.report() == {"rows": {"train": 140, "valid": 30, "test": 30}, "features": 4, "targets": ["price"],
                             "classes": None, "columns": {name: ["StandardScaler"] for name in data.feature_names},
                             "windows": None}
    assert data.rows("train").tolist() == fitted_housing(path).rows("train").tolist()


def test_preprocessors():
    values = numpy.array([[-3.0], [0.0], [7.0]])
    assert pre.Abs().fit(values).transform(values).tolist() == [[3.0], [0.0], [7.0]]
    log = pre.Log(base=2.0, norm=2.0)
    encoded = log.fit(values[1:]).transform(values[1:])
    numpy.testing.assert_allclose(encoded, numpy.log1p(values[1:] / 2.0) / numpy.log(2.0))
    numpy.testing.assert_allclose(log.inverse_transform(encoded), values[1:])
    cast = pre.Cast("int64")
    assert cast.transform(values).dtype == numpy.int64 and cast.rescales is False
    label = pre.Label().fit(numpy.array([["b"], ["a"], ["b"]]))
    assert label.classes_.tolist() == ["a", "b"]
    assert label.transform(numpy.array(["b", "a"])).tolist() == [1, 0]
    assert label.inverse_transform(numpy.array([1, 0])).tolist() == ["b", "a"]
    encoder = pre.one_hot()
    assert encoder.handle_unknown == "ignore" and encoder.sparse_output is False
    with pytest.raises(BuildError):
        pre.Log(base=1.0)
    with pytest.raises(BuildError):
        pre.Cast("not_a_dtype")


def test_aliases_registered():
    names = {name for name, _, _, _ in table.REGISTRATIONS}
    assert names == {"table", "random", "sequential", "kfold", "given", "standard_scaler", "minmax_scaler",
                     "robust_scaler", "one_hot", "label", "abs", "log", "cast"}
    for name, target, slot, _ in table.REGISTRATIONS:
        resolved = aliases.resolve(name)
        found = aliases.import_target(resolved) if isinstance(resolved, str) else resolved
        assert found is table.target_of(name)
        assert aliases.ALIASES[name].slot == slot and aliases.ALIASES[name].doc
    assert table.target_of("table") is table.Table and table.target_of("one_hot") is pre.one_hot
    assert table.target_of("kfold") is splits.KFoldSplit
    assert table.target_of("sklearn.preprocessing.RobustScaler").__name__ == "RobustScaler"
    with pytest.raises(BuildError, match=r"^targets\.price: 'standrad_scaler' .*did you mean standard_scaler"):
        table.chain_of("standrad_scaler", "targets.price")
    with pytest.raises(BuildError, match=r"^split: 'sequentail' .*did you mean sequential"):
        table.split_of("sequentail")
