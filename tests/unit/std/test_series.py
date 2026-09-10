"""Config 03's legos: the sequential split, the window feed, the GRU layers and the forecast plot; kfold too."""

import numpy
import pandas
import pytest
import torch

import kalfa  # noqa: F401
from kalfa.std.feed import WindowDataset, previous_frames, window
from kalfa.std.layer import gru, last_step
from kalfa.std.plot import forecast_samples
from kalfa.std.pre import Frame, apply, fit, standard_scaler
from kalfa.std.split import kfold, kfold_sizes, sequential
from kalfa.synthetic import energy_frame


def test_sequential_split_cuts_every_group_in_order():
    data = energy_frame(sites=2, steps=100)
    parts = sequential(data, [0.7, 0.2, 0.1], group="site_id")
    assert [len(parts[name]) for name in ("train", "valid", "test")] == [140, 40, 20]
    first_site = data[data["site_id"] == "site_0"]
    assert parts["train"][parts["train"]["site_id"] == "site_0"].index.tolist() == first_site.index[:70].tolist()
    assert parts["test"].index.tolist() == first_site.index[90:].tolist() + \
        data[data["site_id"] == "site_1"].index[90:].tolist()
    whole = sequential(data.iloc[:10], [0.5, 0.5, 0.0])
    assert len(whole["train"]) == 5 and len(whole["test"]) == 0


def frames_of(data, group="site_id"):
    prep = fit(data, {"x*": {"preprocessors": ["s"]}, "load": {"target": True}}, {"s": standard_scaler()}, [])
    parts = sequential(data, [0.6, 0.2, 0.2], group=group)
    return prep, {name: apply(parts[name], prep, name) for name in parts}


def test_window_dataset_keeps_series_apart_and_takes_context():
    data = energy_frame(sites=2, steps=50)
    prep, frames = frames_of(data)
    assert frames["train"].extra is not None and list(frames["train"].extra.columns) == ["site_id"]
    train = window(frames["train"], frames, size=5, horizon=3, context=True, group="site_id")
    assert len(train) == 2 * (30 - 5 - 3 + 1)
    assert train.x.shape == (len(train), 5, 3) and train.fields["load"].shape == (len(train), 3)
    valid = window(frames["valid"], frames, size=5, horizon=3, context=True, group="site_id")
    assert len(valid) == 2 * (10 - 3 + 1)
    assert valid.rows()[0] == frames["valid"].index[0]
    plain = window(frames["valid"], frames, size=5, horizon=3, context=False, group="site_id")
    assert len(plain) == 2 * (10 - 5 - 3 + 1)
    item = valid[0]
    assert set(item) == {"x", "load"} and item["x"].shape == (5, 3) and item["load"].shape == (3,)
    first_row = frames["valid"].index[0]
    expected = data.loc[first_row:first_row + 2, "load"].to_numpy()
    assert numpy.allclose(item["load"].numpy(), expected.astype("float32"))
    previous = previous_frames(frames["test"], frames)
    assert previous == [frames["train"], frames["valid"]]
    assert previous_frames(frames["train"], frames) == []
    test = window(frames["test"], frames, size=25, horizon=3, context=True, group="site_id")
    assert len(test) == 2 * (10 - 3 + 1)
    grouped = window(frames["train"], None, size=5, horizon=1, context=True, group="site_id")
    assert len(grouped) == 2 * (30 - 5)
    assert all(frames["train"].extra["site_id"].loc[row] == frames["train"].extra["site_id"].loc[row]
               for row in grouped.rows())


def test_window_never_crosses_a_group_boundary():
    data = pandas.DataFrame({"g": ["a"] * 6 + ["b"] * 6, "x0": numpy.arange(12, dtype="float32"),
                             "load": numpy.arange(12, dtype="float32") * 10})
    frame = Frame(data[["x0", "load"]], ["x0"], {"load": ["load"]}, "train", data[["g"]])
    dataset = WindowDataset(frame, size=3, horizon=2, group="g")
    assert len(dataset) == 2 * (6 - 3 - 2 + 1)
    for position in range(len(dataset)):
        steps = dataset.x[position, :, 0].tolist()
        assert steps == sorted(steps) and (max(steps) < 6 or min(steps) >= 6)
    assert dataset.rows().tolist() == [3, 4, 9, 10]


def test_gru_and_last_step_are_lazy_and_seeded():
    from helpers import linear_graph
    from cirak.build import Graph, GraphNode
    from kalfa.std.builder import Module

    def build(seed):
        graph = Graph(("x",), ("y",), (GraphNode("g", gru(4), ("x",), ("h",)), GraphNode("l", last_step(), ("h",), ("s",)),
                                       GraphNode("o", torch.nn.LazyLinear(2), ("s",), ("y",))))
        return Module(graph, seed=seed)

    first, second = build(3), build(3)
    x = torch.randn(2, 7, 3)
    assert first(x).shape == (2, 2) and torch.equal(first(x), second(x))
    assert first.nodes["g"].core.input_size == 3


def test_forecast_samples_writes_a_file(tmp_path):
    predictions = pandas.DataFrame({"row": [0, 1, 2], **{f"load_{i}": [1.0, 2.0, 3.0] for i in range(4)},
                                    **{f"pred_y_{i}": [1.1, 2.1, 3.1] for i in range(4)}})
    forecast_samples(predictions, [], {}, str(tmp_path), n=2)
    assert (tmp_path / "plots" / "forecast_samples.png").exists()
    assert forecast_samples(pandas.DataFrame(), [], {}, str(tmp_path)) is None


def test_kfold_holds_out_the_fold_and_carves_valid():
    data = energy_frame(sites=1, steps=100)
    parts = kfold(data, k=5, fold=2, val=0.25, seed=1)
    assert [len(parts[name]) for name in ("train", "valid", "test")] == [60, 20, 20]
    again = kfold(data, k=5, fold=2, val=0.25, seed=1)
    assert parts["test"].index.tolist() == again["test"].index.tolist()
    other = kfold(data, k=5, fold=3, val=0.25, seed=1)
    assert not set(parts["test"].index) & set(other["test"].index)
    no_valid = kfold(data, k=5, fold=0, seed=1)
    assert len(no_valid["valid"]) == 0 and len(no_valid["train"]) == 80
    assert kfold_sizes(100, {"k": 5, "fold": 0, "val": 0.25}) == {"train": 60, "valid": 20, "test": 20}
    with pytest.raises(ValueError):
        kfold(data, k=5, fold=7, seed=1)
