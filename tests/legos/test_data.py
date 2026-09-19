import re

import numpy
import pandas
import pytest
import torch
from cirak.registry import registry

from helpers import build, frame
from kalfa.std import STD_URIS
from kalfa.std.pre.base import TableFrame


DATA = ["/data/kalfa/class_weights", "/data/kalfa/feature_index", "/data/kalfa/feature_width",
        "/data/kalfa/target_weights", "/data/kalfa/vocab_size"]
FEATURES = ["num_0", "num_1", "region_east", "region_north", "region_south", "tier"]


def train_table():
    return pandas.DataFrame({"num_0": [1.0, 2.0, 3.0, 4.0], "num_1": [10.0, 20.0, 30.0, 40.0],
                             "region": ["north", "south", "north", "east"], "tier": ["low", "mid", "high", "low"],
                             "y_a": [1.0, 2.0, 3.0, 4.0], "y_b": [0.5, 1.5, 2.5, 3.5], "is_hot": [0, 1, 1, 1]})


def fitted(size=2):
    preprocessors = {"std": build("/pre/sklearn/standard_scaler"), "onehot": build("/pre/kalfa/one_hot"),
                     "ordinal": build("/pre/kalfa/label_encoder")}
    fields = {"num_*": {"preprocessors": ["std"]}, "region": {"preprocessors": ["onehot"]},
              "tier": {"preprocessors": ["ordinal"]}, "y_*": {"target": True, "preprocessors": ["std"]},
              "is_hot": {"target": True}}
    prep = build("/lego/kalfa/fit", df=train_table(), fields=fields, preprocessors=preprocessors, drop=[])
    table = build("/lego/kalfa/apply", df=train_table(), prep=prep, set="train")
    data = build("/feed/kalfa/table", frame=table, frames={"train": table})
    return prep, build("/loader/kalfa/torch", data=data, set="train", size=size, shuffle=False)


def labelled(labels):
    data = pandas.DataFrame({"x0": numpy.arange(len(labels), dtype="float32"),
                             "label": numpy.asarray(labels, dtype="int64")})
    table = TableFrame(["x0"], {"label": ["label"]}, "train", data=data)
    dataset = build("/feed/kalfa/table", frame=table, frames={"train": table})
    return build("/loader/kalfa/torch", data=dataset, set="train", size=4)


def test_data_catalog_aliases_and_facts():
    assert sorted(uri for uri in STD_URIS if uri.startswith("/data/")) == DATA
    for uri in DATA:
        assert registry.aliases()[uri.rsplit("/", 1)[1]] == uri
        assert registry.facts(uri).get("counts") is (True if uri == "/data/kalfa/class_weights" else None)


def test_feature_width_is_the_width_of_x_from_the_fitted_plan():
    prep, loader = fitted()
    assert prep.features == FEATURES
    assert build("/data/kalfa/feature_width", loader=loader) == 6
    assert next(iter(loader))["x"].shape == (2, 6)
    plain = frame(rows=4, features=3)
    dataset = build("/feed/kalfa/table", frame=plain, frames={"train": plain})
    assert build("/data/kalfa/feature_width", loader=build("/loader/kalfa/torch", data=dataset, set="train")) == 3


def test_feature_index_follows_the_fitted_plan_with_a_widened_column():
    _, loader = fitted()
    assert build("/data/kalfa/feature_index", loader=loader, columns=["num_1", "region_*"]) == [1, 2, 3, 4]
    assert build("/data/kalfa/feature_index", loader=loader, columns="tier") == [5]
    assert build("/data/kalfa/feature_index", loader=loader, columns="region_north") == [3]
    assert build("/data/kalfa/feature_index", loader=loader, columns=["region_north", "num_*"]) == [3, 0, 1]
    assert build("/data/kalfa/feature_index", loader=loader, columns=["num_*", "num_0"]) == [0, 1]
    assert build("/data/kalfa/feature_index", loader=loader, columns=["*"]) == [0, 1, 2, 3, 4, 5]
    with pytest.raises(ValueError, match=re.escape("feature_index: 'zzz' matches no feature column; the columns are "
                                                   "['num_0', 'num_1', 'region_east', 'region_north', "
                                                   "'region_south', 'tier']")):
        build("/data/kalfa/feature_index", loader=loader, columns=["num_0", "zzz"])
    with pytest.raises(ValueError, match=re.escape("feature_index: 'region' matches no feature column")):
        build("/data/kalfa/feature_index", loader=loader, columns="region")


def test_class_weights_are_inverse_frequencies_with_weighted_mean_one():
    weights = build("/data/kalfa/class_weights", loader=labelled([0, 1, 1, 1]))
    assert weights.dtype == torch.float32
    torch.testing.assert_close(weights, torch.tensor([2.0, 2.0 / 3.0]))
    counts = torch.tensor([1.0, 3.0])
    assert (counts * weights).sum().item() / 4.0 == pytest.approx(1.0)
    torch.testing.assert_close(build("/data/kalfa/class_weights", loader=labelled([0, 2, 2])),
                               torch.tensor([1.0, 1.0, 0.5]))
    torch.testing.assert_close(build("/data/kalfa/class_weights", loader=labelled([1, 1, 0, 0])),
                               torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(build("/data/kalfa/class_weights", loader=labelled([0, 1, 1, 1]), target="label"),
                               torch.tensor([2.0, 2.0 / 3.0]))


def test_class_weights_name_the_target_among_several_and_need_a_table_dataset(root):
    _, loader = fitted()
    torch.testing.assert_close(build("/data/kalfa/class_weights", loader=loader, target="is_hot"),
                               torch.tensor([2.0, 2.0 / 3.0]))
    with pytest.raises(ValueError, match=re.escape("class_weights needs one target field or a target name; the "
                                                   "dataset has ['y_a', 'y_b', 'is_hot']")):
        build("/data/kalfa/class_weights", loader=loader)
    stream = build("/source/kalfa/parquet_stream", path=str(root / "housing.parquet"))
    prep = build("/lego/kalfa/fit", df=stream, fields={"x0": {}, "price": {"target": True}}, preprocessors={},
                 drop=[])
    view = build("/lego/kalfa/apply", df=stream, prep=prep, set="train")
    lazy = build("/loader/kalfa/torch", data=build("/feed/kalfa/table", frame=view, frames={}), set="train", size=8)
    with pytest.raises(ValueError, match=re.escape("a lazy set cannot be counted: balanced and class_weights need a "
                                                   "table source")):
        build("/data/kalfa/class_weights", loader=lazy)
    series = frame(rows=8, features=2)
    windows = build("/feed/kalfa/window", frame=series, frames={"train": series}, size=2, horizon=1)
    with pytest.raises(ValueError, match=re.escape("WindowDataset cannot count its labels; balanced and "
                                                   "class_weights need a table dataset")):
        build("/data/kalfa/class_weights", loader=build("/loader/kalfa/torch", data=windows, set="train", size=4))


def test_vocab_size_is_the_size_of_the_fitted_tokenizer(root):
    lines = build("/source/kalfa/text_lines", path=str(root / "text.txt"))
    prep = build("/lego/kalfa/fit", df=lines, fields={"text": {"preprocessors": ["tok"]}},
                 preprocessors={"tok": build("/pre/kalfa/char_tokenizer")}, drop=[])
    text = (root / "text.txt").read_text(encoding="utf-8")
    size = build("/data/kalfa/vocab_size", prep=prep)
    assert isinstance(size, int)
    assert size == len(set(text) | {"\n"})
    assert size == prep.tokenizer().size
    message = "vocab_size needs a fitted tokenizer among the preprocessors (char_tokenizer)"
    with pytest.raises(ValueError, match=re.escape(message)):
        build("/data/kalfa/vocab_size", prep=fitted()[0])
    with pytest.raises(ValueError, match=re.escape(message)):
        build("/data/kalfa/vocab_size", prep=None)


def test_target_weights_resolve_names_and_globs_against_the_target_columns():
    _, loader = fitted()
    assert loader.dataset.frame.targets == {"y_a": ["y_a"], "y_b": ["y_b"], "is_hot": ["is_hot"]}
    weights = build("/data/kalfa/target_weights", loader=loader, weights={"y_*": 2.0, "default": 0.5})
    assert weights.dtype == torch.float32
    assert weights.tolist() == [2.0, 2.0, 0.5]
    assert build("/data/kalfa/target_weights", loader=loader, weights={"y_*": 2.0, "default": 0.5},
                 target="is_hot").tolist() == [0.5]
    assert build("/data/kalfa/target_weights", loader=loader, weights={"y_*": 2.0}, target="y_a").tolist() == [2.0]
    assert build("/data/kalfa/target_weights", loader=loader, weights=None).tolist() == [1.0, 1.0, 1.0]
    assert build("/data/kalfa/target_weights", loader=loader, weights={"y_b": 3}).tolist() == [1.0, 3.0, 1.0]
    assert build("/data/kalfa/target_weights", loader=loader, weights={"y_a": 5.0, "y_*": 2.0}).tolist() == [5.0, 2.0,
                                                                                                             1.0]
    with pytest.raises(ValueError, match=re.escape("target_weights names target 'w'; the dataset has ['is_hot', "
                                                   "'y_a', 'y_b']")):
        build("/data/kalfa/target_weights", loader=loader, weights={}, target="w")
