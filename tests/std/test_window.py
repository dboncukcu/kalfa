import numpy
import pandas
import pytest
import torch
from torch.utils.data import DataLoader

from kalfa.core import aliases
from kalfa.core.errors import BuildError
from kalfa.std.data import table
from kalfa.std.data.window import Window, WindowDataset


class FakeRng:
    def stream(self, name, device=None):
        return torch.Generator().manual_seed(11)


class FakeContext:
    def __init__(self, seed=7):
        self.seed = seed
        self.rng = FakeRng()


def make(ids):
    ids = torch.as_tensor(list(ids), dtype=torch.float32)
    x = torch.stack([ids, ids + 0.5], dim=1)
    targets = {"y": (10 * ids).reshape(-1, 1), "z": torch.stack([ids, -ids], dim=1),
               "k": ids.to(torch.int64) % 3}
    return x, targets, 100 + ids.numpy().astype("int64")


def ids_of(item):
    return item["x"][:, 0].to(torch.int64).tolist()


def test_windows_slice_inputs_and_targets():
    x, targets, rows = make(range(10))
    dataset = Window(3).dataset(x, targets, rows)
    assert isinstance(dataset, WindowDataset) and len(dataset) == 7
    item = dataset[0]
    assert set(item) == {"x", "y", "z", "k"}
    assert item["x"].dtype == torch.float32 and tuple(item["x"].shape) == (3, 2)
    assert torch.equal(item["x"], x[0:3])
    assert tuple(item["y"].shape) == (1,) and item["y"].tolist() == [30.0]
    assert tuple(item["z"].shape) == (1, 2) and item["z"].tolist() == [[3.0, -3.0]]
    assert tuple(item["k"].shape) == (1,) and item["k"].dtype == torch.int64 and item["k"].tolist() == [0]
    assert ids_of(dataset[6]) == [6, 7, 8] and dataset[6]["y"].tolist() == [90.0]
    assert isinstance(dataset.rows, numpy.ndarray) and dataset.rows.tolist() == list(range(103, 110))


def test_horizon_three_and_two_columns():
    x, targets, rows = make(range(10))
    dataset = Window(3, horizon=3).dataset(x, targets, rows)
    assert len(dataset) == 5
    item = dataset[4]
    assert ids_of(item) == [4, 5, 6]
    assert tuple(item["y"].shape) == (3,) and item["y"].tolist() == [70.0, 80.0, 90.0]
    assert tuple(item["z"].shape) == (3, 2) and item["z"][:, 0].tolist() == [7.0, 8.0, 9.0]
    assert tuple(item["k"].shape) == (3,) and item["k"].tolist() == [1, 2, 0]
    assert dataset.rows.tolist() == [103, 104, 105, 106, 107]


def test_stride_skips_starts():
    x, targets, rows = make(range(10))
    dataset = Window(2, stride=3).dataset(x, targets, rows)
    assert len(dataset) == 3
    assert dataset.starts.tolist() == [0, 3, 6]
    assert [ids_of(dataset[position]) for position in range(3)] == [[0, 1], [3, 4], [6, 7]]
    assert dataset.rows.tolist() == [102, 105, 108]
    wide = Window(3, horizon=2, stride=4).dataset(x, targets, rows)
    assert wide.rows.tolist() == [103, 107]


def test_windows_stay_in_one_group():
    x, targets, rows = make(range(20))
    groups = numpy.array(["a"] * 12 + ["b"] * 8)
    dataset = Window(3, horizon=2, group="g").dataset(x, targets, rows, groups)
    assert len(dataset) == 12
    assert dataset.rows.tolist() == list(range(103, 111)) + list(range(115, 119))
    for position in range(len(dataset)):
        item = dataset[position]
        inputs = ids_of(item)
        outputs = (item["y"] / 10).to(torch.int64).tolist()
        assert inputs + outputs == list(range(inputs[0], inputs[0] + 5))
        assert len({groups[index] for index in inputs + outputs}) == 1
        assert dataset.rows[position] == 100 + outputs[0]


def test_context_reaches_back_into_previous_set():
    before = make(range(10))
    x, targets, rows = make(range(10, 16))
    dataset = Window(4).dataset(x, targets, rows, None, (*before, None))
    assert len(dataset) == 6
    assert ids_of(dataset[0]) == [6, 7, 8, 9] and dataset[0]["y"].tolist() == [100.0]
    assert ids_of(dataset[1]) == [7, 8, 9, 10]
    assert dataset.rows.tolist() == list(range(110, 116))
    alone = Window(4).dataset(x, targets, rows)
    assert alone.rows.tolist() == [114, 115]
    ignored = Window(4, context=False).dataset(x, targets, rows, None, (*before, None))
    assert ignored.rows.tolist() == [114, 115]
    longer = Window(4, horizon=3).dataset(x, targets, rows, None, (*before, None))
    assert longer.rows.tolist() == [110, 111, 112, 113]
    assert longer[3]["y"].tolist() == [130.0, 140.0, 150.0]


def test_context_follows_each_group():
    before = make([*range(0, 6), *range(20, 26)])
    earlier_groups = numpy.array(["a"] * 6 + ["b"] * 6)
    x, targets, rows = make([*range(6, 10), *range(26, 30)])
    groups = numpy.array(["a"] * 4 + ["b"] * 4)
    dataset = Window(3, group="g").dataset(x, targets, rows, groups, (*before, earlier_groups))
    assert len(dataset) == 8
    assert dataset.rows.tolist() == [106, 107, 108, 109, 126, 127, 128, 129]
    assert ids_of(dataset[0]) == [3, 4, 5] and dataset[0]["y"].tolist() == [60.0]
    assert ids_of(dataset[4]) == [23, 24, 25] and dataset[4]["y"].tolist() == [260.0]
    assert ids_of(dataset[7]) == [26, 27, 28]
    stranger = numpy.array(["c"] * 6 + ["b"] * 6)
    partial = Window(3, group="g").dataset(x, targets, rows, groups, (*before, stranger))
    assert partial.rows.tolist() == [109, 126, 127, 128, 129]


def test_short_parts():
    x, targets, rows = make([10, 11])
    empty = Window(4).dataset(x, targets, rows)
    assert len(empty) == 0 and empty.rows.tolist() == [] and tuple(empty.x.shape) == (0, 2)
    assert list(DataLoader(empty, batch_size=4)) == []
    reached = Window(4).dataset(x, targets, rows, None, (*make(range(6)), None))
    assert reached.rows.tolist() == [110, 111]
    assert ids_of(reached[0]) == [2, 3, 4, 5] and ids_of(reached[1]) == [3, 4, 5, 10]
    x, targets, rows = make([2, 3, 4])
    stretched = Window(4).dataset(x, targets, rows, None, (*make([0, 1]), None))
    assert len(stretched) == 1 and ids_of(stretched[0]) == [0, 1, 2, 3] and stretched[0]["y"].tolist() == [40.0]
    assert stretched.rows.tolist() == [104]
    x, targets, rows = make(range(5))
    assert len(Window(3, horizon=3).dataset(x, targets, rows)) == 0


def test_interleaved_groups_form_one_series_each():
    x, targets, rows = make(range(12))
    groups = numpy.array(["a", "b"] * 6)
    dataset = Window(2, group="g").dataset(x, targets, rows, groups)
    assert dataset.rows.tolist() == list(range(104, 112))
    assert ids_of(dataset[0]) == [0, 2] and dataset[0]["y"].tolist() == [40.0]
    assert ids_of(dataset[1]) == [1, 3] and dataset[1]["y"].tolist() == [50.0]


def test_targets_may_be_missing():
    x, _, rows = make(range(8))
    dataset = Window(3, horizon=2).dataset(x, {}, rows)
    assert len(dataset) == 4 and set(dataset[0]) == {"x"}


def test_loader_batches_windows():
    x, targets, rows = make(range(20))
    batch = next(iter(DataLoader(Window(3).dataset(x, targets, rows), batch_size=4)))
    assert tuple(batch["x"].shape) == (4, 3, 2) and batch["x"].dtype == torch.float32
    assert tuple(batch["y"].shape) == (4, 1) and tuple(batch["z"].shape) == (4, 1, 2)
    assert tuple(batch["k"].shape) == (4, 1) and batch["k"].dtype == torch.int64
    batch = next(iter(DataLoader(Window(3, horizon=3).dataset(x, targets, rows), batch_size=4)))
    assert tuple(batch["y"].shape) == (4, 3) and tuple(batch["z"].shape) == (4, 3, 2)


def test_bad_parameters():
    for params in ({"size": 0}, {"size": 3, "horizon": 0}, {"size": 3, "stride": 0}, {"size": True},
                   {"size": 2.5}):
        with pytest.raises(BuildError, match="positive integer"):
            Window(**params)
    x, targets, rows = make(range(6))
    with pytest.raises(BuildError, match="row ids"):
        Window(2).dataset(x, targets, rows[:3])
    with pytest.raises(BuildError, match="group values"):
        Window(2, group="g").dataset(x, targets, rows, numpy.array(["a"] * 4))
    with pytest.raises(BuildError, match="context rows"):
        Window(2).dataset(x, targets, rows, None, (torch.zeros(3, 5), {}, numpy.arange(3), None))


def test_alias_registered():
    assert aliases.resolve("window") == "kalfa.std.data.window.Window"
    assert aliases.ALIASES["window"].slot == "value" and table.target_of("window") is Window


def series_frame(count=120):
    rng = numpy.random.default_rng(3)
    pieces = []
    for name in ("a", "b"):
        steps = numpy.arange(count)
        level = numpy.sin(steps / 6.0) + rng.normal(scale=0.05, size=count)
        pieces.append(pandas.DataFrame({"f1": level, "f2": numpy.cos(steps / 6.0), "y": numpy.roll(level, -1),
                                        "g": name}))
    return pandas.concat(pieces, ignore_index=True)


def test_table_feeds_windows(tmp_path):
    frame = series_frame()
    path = tmp_path / "series.parquet"
    frame.to_parquet(path)
    data = table.Table(str(path), split={"use": "sequential", "ratios": [0.6, 0.2, 0.2], "group": "g"},
                       features={"f*": None}, targets={"y": None}, feed=Window(5, horizon=2, group="g"), batch=16)
    data.fit(FakeContext())
    assert data.feature_names == ["f1", "f2"]
    assert {name: len(dataset) for name, dataset in data.datasets.items()} == {"train": 132, "valid": 46,
                                                                                "test": 46}
    loaders = data.loaders()
    batch = next(iter(loaders["train"]))
    assert set(batch) == {"x", "y"}
    assert tuple(batch["x"].shape) == (16, 5, 2) and batch["x"].dtype == torch.float32
    assert tuple(batch["y"].shape) == (16, 2) and batch["y"].dtype == torch.float32
    test = data.datasets["test"]
    rows = data.rows("test")
    assert numpy.array_equal(rows, test.rows)
    assert rows.tolist() == list(range(96, 119)) + list(range(216, 239))
    features = frame[["f1", "f2"]].to_numpy(dtype="float32")
    targets = frame["y"].to_numpy(dtype="float32")
    for position in (0, 1, 22, 23, 45):
        first = int(rows[position])
        item = test[position]
        numpy.testing.assert_allclose(item["x"].numpy(), features[first - 5:first])
        numpy.testing.assert_allclose(item["y"].numpy(), targets[first:first + 2])
    collected = numpy.concatenate([piece["y"].numpy() for piece in loaders["test"]])
    assert collected.shape == (46, 2)
    numpy.testing.assert_allclose(collected[:, 0], targets[rows])
    assert data.rows("valid").tolist() == list(range(72, 95)) + list(range(192, 215))
    fresh = frame.iloc[100:110]
    loader = data.loader_for(fresh)
    assert loader.dataset.rows.tolist() == list(range(105, 109))
    assert tuple(next(iter(loader))["x"].shape) == (4, 5, 2)
