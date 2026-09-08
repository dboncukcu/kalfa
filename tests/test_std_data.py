"""Data legos: sources, filters, the random split, fit and apply, the table feed and the loader."""

import numpy
import pandas
import pytest
import torch

import kalfa  # noqa: F401
from helpers import frame
from kalfa.std.data import filter as filter_rows
from kalfa.std.data import filter_set
from kalfa.std.feed import table
from kalfa.std.loader import torch as torch_loader
from kalfa.std.pre import (Prep, apply, assign_fields, fit, minmax_scaler, read_prep, specificity, standard_scaler,
                           torch_dtype, write_prep)
from kalfa.std.source import csv, header, parquet
from kalfa.std.split import random as random_split
from kalfa.std.split import sizes
from kalfa.synthetic import housing_frame


def test_sources_and_headers(tmp_path):
    data = housing_frame(rows=20)
    data.to_parquet(tmp_path / "h.parquet", index=False)
    data.to_csv(tmp_path / "h.csv", index=False)
    assert len(parquet(str(tmp_path / "h.parquet"))) == 20
    assert len(csv(str(tmp_path / "h.csv"))) == 20
    head = header("/source/kalfa/parquet", {"path": str(tmp_path / "h.parquet")})
    assert head["rows"] == 20 and head["columns"][-1] == "price" and head["dtypes"]["price"] == "double"
    head = header("/source/kalfa/csv", {"path": str(tmp_path / "h.csv")})
    assert head["rows"] == 20 and "x0" in head["columns"]
    assert header("/source/other/x", {}) is None


def test_filters():
    data = pandas.DataFrame({"a": [1, 2, 3, 4], "b": [0, 1, 0, 1]})
    assert filter_rows(data, "a > 2")["a"].tolist() == [3, 4]
    filters = [{"query": "b == 1", "sets": ["train"]}, {"query": "a < 4", "sets": ["train", "valid"]}]
    assert filter_set(data, "train", filters)["a"].tolist() == [2]
    assert filter_set(data, "valid", filters)["a"].tolist() == [1, 2, 3]
    assert filter_set(data, "test", filters)["a"].tolist() == [1, 2, 3, 4]


def test_random_split_is_seeded_and_an_empty_ratio_gives_an_empty_frame():
    data = housing_frame(rows=100)
    first = random_split(data, [0.7, 0.15, 0.15], seed=3)
    second = random_split(data, [0.7, 0.15, 0.15], seed=3)
    assert [len(first[name]) for name in ("train", "valid", "test")] == [70, 15, 15]
    assert first["train"].index.tolist() == second["train"].index.tolist()
    assert set(first["train"].index) | set(first["valid"].index) | set(first["test"].index) == set(range(100))
    none = random_split(data, [0.8, 0.0, 0.2], seed=1)
    assert len(none["valid"]) == 0 and list(none["valid"].columns) == list(data.columns)
    assert sizes(100, [0.8, 0.0, 0.2]) == {"train": 80, "valid": 0, "test": 20}
    with pytest.raises(ValueError):
        random_split(data, [0.5, 0.5], seed=1)


def test_glob_specificity_rules():
    columns = ["x1", "x2", "xx1", "price", "input"]
    owners, problems = assign_fields(columns, ["x*", "x?", "xx1", "price"])
    assert problems == []
    assert owners["x1"] == "x?" and owners["x2"] == "x?" and owners["xx1"] == "xx1" and owners["price"] == "price"
    owners, problems = assign_fields(["ab"], ["a?", "?b"])
    assert [kind for kind, _ in problems] == ["glob_ambiguous"]
    owners, problems = assign_fields(["ab"], ["z*"])
    assert [kind for kind, _ in problems] == ["column_missing"]
    assert specificity("x??") == (1, 2) and specificity("xy*") == (2, 0)


def test_dtype_table():
    assert torch_dtype("float64") == "float32" and torch_dtype("float16") == "float32"
    assert torch_dtype("int8") == "int64" and torch_dtype("uint8") == "int64" and torch_dtype("Int64") == "int64"
    assert torch_dtype("bool") == "bool"
    assert torch_dtype("object") is None and torch_dtype("category") is None and torch_dtype("datetime64[ns]") is None


def test_a_grouped_preprocessor_is_one_object_over_all_its_columns(tmp_path):
    data = housing_frame(rows=200)
    prep = fit(data, {"x*": {"preprocessors": ["s"]}, "price": {"target": True, "preprocessors": ["s"]}},
               {"s": standard_scaler()}, [], record=str(tmp_path))
    grouped = prep.fitted["s"]
    assert grouped.columns == [f"x{position}" for position in range(8)] + ["price"]
    assert grouped.obj.scaler.mean_.shape == (9,)
    assert float(grouped.obj.scaler.mean_[-1]) == pytest.approx(float(data["price"].mean()))
    train = apply(data, prep, "train")
    assert abs(float(train.data["price"].mean())) < 1e-6 and abs(float(train.data["x3"].mean())) < 1e-6
    restored = prep.inverse("price", train.data["price"].to_numpy())
    assert numpy.allclose(restored, data["price"].to_numpy(), atol=1e-3)
    again = read_prep(str(tmp_path))
    assert again.fitted["s"].columns == grouped.columns
    assert numpy.allclose(again.object_of("s", "x2").apply([0.0, 1.0]),
                          prep.object_of("s", "x2").apply([0.0, 1.0]))


def test_a_grouped_preprocessor_waits_for_the_per_column_steps_before_it(tmp_path):
    from kalfa.std.pre import cast

    data = housing_frame(rows=50)
    prep = fit(data, {"x*": {"preprocessors": ["c", "s"]}, "price": {"target": True}},
               {"c": cast("float32"), "s": standard_scaler()}, [])
    assert sorted(prep.fitted["c"]) == [f"x{position}" for position in range(8)]
    assert prep.fitted["s"].columns == [f"x{position}" for position in range(8)]
    train = apply(data, prep, "train")
    assert abs(float(train.data["x5"].mean())) < 1e-5


def test_the_grouped_fact_matches_the_object_the_lego_builds():
    from cirak.registry import registry

    from kalfa.std.pre import is_grouped

    for uri in sorted(registry.uris()):
        if not uri.startswith("/pre/"):
            continue
        entry = registry.lookup(uri)
        if entry.facts.partial:
            continue
        try:
            built = registry.resolve(uri)()
        except TypeError:                                   # the lego needs params (cast, resize, normalize)
            continue
        assert is_grouped(built) == registry.facts(uri).get("grouped", False), uri


def test_grouped_preprocessors_written_in_different_orders_are_an_error():
    data = housing_frame(rows=40)
    fields = {"x0": {"preprocessors": ["a", "b"]}, "x1": {"preprocessors": ["b", "a"]},
              "x*": {"preprocessors": ["a"]}, "price": {"target": True}}
    with pytest.raises(ValueError, match="different orders"):
        fit(data, fields, {"a": standard_scaler(), "b": minmax_scaler()}, [])


def test_a_grouped_preprocessor_needs_one_column_wide_input():
    from kalfa.std.pre import one_hot

    data = housing_frame(rows=40)
    data["kind"] = ["a", "b"] * 20
    fields = {"kind": {"preprocessors": ["hot", "s"]}, "x*": {}, "price": {"target": True}}
    with pytest.raises(ValueError, match="one column wide"):
        fit(data, fields, {"hot": one_hot(), "s": standard_scaler()}, [])


def test_fit_and_apply_scale_features_and_invert_the_target(tmp_path):
    data = housing_frame(rows=200)
    scaler = standard_scaler()
    prep = fit(data, {"x*": {"preprocessors": ["s"]}, "price": {"target": True, "preprocessors": ["t"]}},
               {"s": scaler, "t": standard_scaler()}, [], record=str(tmp_path))
    assert prep.features == [f"x{i}" for i in range(8)] and prep.targets == {"price": ["price"]}
    assert set(prep.fitted["s"].columns) == set(prep.features) and set(prep.fitted["t"].columns) == {"price"}
    train = apply(data, prep, "train")
    assert abs(float(train.data["x0"].mean())) < 1e-5 and abs(float(train.data["price"].std()) - 1.0) < 0.01
    assert train.data.dtypes["x0"] == "float32"
    restored = prep.inverse("price", train.data["price"].to_numpy())
    assert numpy.allclose(restored, data["price"].to_numpy(), atol=1e-3)
    assert (tmp_path / "preprocessors" / "s.pkl").exists() and (tmp_path / "preprocessors" / "plan.json").exists()
    again = read_prep(str(tmp_path))
    assert again.features == prep.features and again.targets == prep.targets
    assert numpy.allclose(apply(data, again, "test").data["x3"].to_numpy(), train.data["x3"].to_numpy())


def test_fit_errors(tmp_path):
    data = housing_frame(rows=20)
    with pytest.raises(ValueError, match="does not define"):
        fit(data, {"x*": {"preprocessors": ["ghost"]}, "price": {"target": True}}, {}, [])
    data["input"] = 1.0
    with pytest.raises(ValueError, match="reserved"):
        fit(data, {"input": {}, "price": {"target": True}}, {}, [])
    data["kind"] = "a"
    with pytest.raises(TypeError, match="dtype"):
        fit(data, {"kind": {}, "price": {"target": True}}, {}, [])


def test_preprocessor_sets_limit_where_a_chain_applies():
    data = housing_frame(rows=30)
    keys = {"s": {"sets": ["train"]}, "t": {}}
    prep = fit(data, {"x0": {"preprocessors": ["s"]}, "price": {"target": True, "preprocessors": ["t"]}},
               {"s": standard_scaler(), "t": standard_scaler()}, [], keys=keys)
    assert prep.sets == {"s": ["train"]}
    train = apply(data, prep, "train", keys)
    valid = apply(data, prep, "valid", keys)
    assert abs(float(train.data["x0"].mean())) < 1e-5
    assert numpy.allclose(valid.data["x0"].to_numpy(), data["x0"].to_numpy(dtype="float32"))
    assert numpy.allclose(apply(data, prep, "valid").data["x0"].to_numpy(), valid.data["x0"].to_numpy())


def test_drop_and_empty_set_keep_the_layout():
    data = housing_frame(rows=20)
    prep = fit(data, {"x*": {}, "price": {"target": True}}, {}, ["x7"])
    assert "x7" not in prep.features and len(prep.features) == 7
    empty = apply(data.iloc[:0], prep, "valid")
    assert len(empty) == 0 and list(empty.data.columns) == prep.features + ["price"]
    dataset = table(empty)
    assert len(dataset) == 0 and dataset.x.shape == (0, 7)


def test_table_feed_names_x_and_targets():
    dataset = table(frame(rows=10, features=3))
    assert dataset.inputs == ["x"] and dataset.targets == ["price"]
    item = dataset[0]
    assert set(item) == {"x", "price"} and item["x"].shape == (3,) and item["x"].dtype == torch.float32
    assert item["price"].shape == () and len(dataset) == 10


def test_loader_shuffles_the_train_set_only():
    dataset = table(frame(rows=20, features=2))
    torch.manual_seed(0)
    train = torch_loader(dataset, "train", {"size": 4})
    first = torch.cat([batch["price"] for batch in train])
    torch.manual_seed(1)
    second = torch.cat([batch["price"] for batch in train])
    assert not torch.equal(first, second)
    valid = torch_loader(dataset, "valid", {"size": 4, "eval_size": 10})
    batches = list(valid)
    assert len(batches) == 2 and torch.equal(batches[0]["price"], dataset.fields["price"][:10])
    assert set(batches[0]) == {"x", "price"}
    assert torch_loader(dataset, "train", {"size": 4, "shuffle": False}).batch_size == 4
