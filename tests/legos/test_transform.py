import functools
import re

import numpy
import pandas
import pytest
from cirak.registry import registry

from helpers import build
from kalfa.std import STD_URIS
from kalfa.std.common.samples import Samples
from kalfa.std.common.stream import Stream


TRANSFORMS = {"filter": "/transform/kalfa/filter", "derive": "/transform/kalfa/derive",
              "rename": "/transform/kalfa/rename", "astype": "/transform/kalfa/astype", "drop": "/transform/kalfa/drop"}
TABLE_ONLY = "needs a table in memory; a stream or a Dataset source keeps its rows as they are and filter is the " \
             "transform they take"


def table():
    return pandas.DataFrame({"raw_0": [1.0, 2.0, 3.0, 4.0], "raw_1": [10.0, 20.0, 30.0, 40.0],
                             "count": [1, 2, 3, 4], "junk": ["a", "b", "a", "b"]}, index=[10, 11, 12, 13])


def test_transform_catalog_is_the_five_table_transforms():
    assert sorted(uri for uri in STD_URIS if uri.startswith("/transform/")) == sorted(TRANSFORMS.values())


@pytest.mark.parametrize("name", sorted(TRANSFORMS))
def test_transform_alias_and_partial_fact(name):
    assert registry.aliases()[name] == TRANSFORMS[name]
    assert registry.facts(TRANSFORMS[name]).partial is True
    assert registry.facts(TRANSFORMS[name]).get("needs_table") is (None if name == "filter" else True)


def test_filter_keeps_the_rows_the_query_selects():
    step = build("/transform/kalfa/filter", query="raw_0 > 2")
    assert isinstance(step, functools.partial)
    assert step.keywords == {"query": "raw_0 > 2"}
    df = table()
    out = step(df)
    assert out.index.tolist() == [12, 13]
    pandas.testing.assert_frame_equal(out, df.iloc[2:])
    assert df.index.tolist() == [10, 11, 12, 13]
    assert step(df.iloc[:0]).empty
    assert build("/transform/kalfa/filter", query="junk == 'a' and count < 3")(df).index.tolist() == [10]


def test_filter_on_a_stream_queues_the_query_chunk_by_chunk(root):
    stream = build("/source/kalfa/parquet_stream", path=str(root / "housing.parquet"), chunk=700)
    out = build("/transform/kalfa/filter", query="price > 200")(stream)
    assert isinstance(out, Stream)
    assert out.queries == ("price > 200",)
    assert out.rows == 2000
    housing = pandas.read_parquet(root / "housing.parquet")
    assert out.count() == int((housing["price"] > 200).sum())
    again = build("/transform/kalfa/filter", query="x0 > 0")(out)
    assert again.queries == ("price > 200", "x0 > 0")
    assert again.count() == int(((housing["price"] > 200) & (housing["x0"] > 0)).sum())


def test_filter_on_a_dataset_source_takes_field_equality(root):
    samples = build("/source/kalfa/image_folder", path=str(root / "images"))
    out = build("/transform/kalfa/filter", query="label == 1")(samples)
    assert isinstance(out, Samples)
    assert len(out) == 24
    numpy.testing.assert_array_equal(out.positions, numpy.arange(40, 64))
    with pytest.raises(ValueError, match="a Dataset source takes field equality filters only"):
        build("/transform/kalfa/filter", query="label > 0")(samples)


def test_derive_adds_a_column_from_an_eval_expression():
    df = table()
    out = build("/transform/kalfa/derive", column="inter", expr="raw_0 * raw_1")(df)
    assert list(out.columns) == ["raw_0", "raw_1", "count", "junk", "inter"]
    assert out["inter"].tolist() == [10.0, 40.0, 90.0, 160.0]
    assert list(df.columns) == ["raw_0", "raw_1", "count", "junk"]
    flag = build("/transform/kalfa/derive", column="big", expr="raw_1 > 25")(df)
    assert flag["big"].tolist() == [False, False, True, True]
    assert str(flag["big"].dtype) == "bool"
    total = build("/transform/kalfa/derive", column="total", expr="raw_0 + count")(df)
    assert total["total"].tolist() == [2.0, 4.0, 6.0, 8.0]
    logged = build("/transform/kalfa/derive", column="lg", expr="log10(raw_1)")(df)
    numpy.testing.assert_allclose(logged["lg"].to_numpy(), numpy.log10([10.0, 20.0, 30.0, 40.0]))
    replaced = build("/transform/kalfa/derive", column="raw_0", expr="raw_0 * 2")(df)
    assert replaced["raw_0"].tolist() == [2.0, 4.0, 6.0, 8.0]
    assert list(replaced.columns) == ["raw_0", "raw_1", "count", "junk"]


def test_rename_rewrites_the_matching_columns_with_backreferences():
    df = table()
    out = build("/transform/kalfa/rename", pattern="^raw_(\\d)$", to="num_\\1")(df)
    assert list(out.columns) == ["num_0", "num_1", "count", "junk"]
    assert out["num_1"].tolist() == [10.0, 20.0, 30.0, 40.0]
    assert list(df.columns) == ["raw_0", "raw_1", "count", "junk"]
    scores = pandas.DataFrame({"cms_pt_Z_score": [1.0], "cms_eta_Z_score": [2.0], "run": [3]})
    out = build("/transform/kalfa/rename", pattern="cms_(.*)_Z_score", to="z_\\1")(scores)
    assert list(out.columns) == ["z_pt", "z_eta", "run"]
    untouched = build("/transform/kalfa/rename", pattern="^zzz$", to="q")(df)
    assert list(untouched.columns) == list(df.columns)


def test_astype_casts_the_named_columns():
    df = table()
    out = build("/transform/kalfa/astype", columns={"count": "float32", "raw_0": "int64"})(df)
    assert str(out["count"].dtype) == "float32"
    assert str(out["raw_0"].dtype) == "int64"
    assert out["count"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert out["raw_0"].tolist() == [1, 2, 3, 4]
    assert str(out["raw_1"].dtype) == "float64"
    assert str(df["count"].dtype) == "int64"
    with pytest.raises(KeyError):
        build("/transform/kalfa/astype", columns={"zzz": "float32"})(df)


def test_drop_removes_the_named_columns():
    df = table()
    out = build("/transform/kalfa/drop", columns=["junk", "count"])(df)
    assert list(out.columns) == ["raw_0", "raw_1"]
    assert list(df.columns) == ["raw_0", "raw_1", "count", "junk"]
    pandas.testing.assert_frame_equal(build("/transform/kalfa/drop", columns=[])(df), df)
    with pytest.raises(KeyError):
        build("/transform/kalfa/drop", columns=["zzz"])(df)


@pytest.mark.parametrize("name, params", [("derive", {"column": "a", "expr": "x0 + 1"}),
                                          ("rename", {"pattern": "x", "to": "y"}),
                                          ("astype", {"columns": {"x0": "float32"}}),
                                          ("drop", {"columns": ["x0"]})])
def test_table_transforms_refuse_a_stream_and_a_dataset_source(root, name, params):
    stream = build("/source/kalfa/parquet_stream", path=str(root / "housing.parquet"))
    with pytest.raises(ValueError, match=re.escape(f"{name} {TABLE_ONLY}")):
        build(TRANSFORMS[name], **params)(stream)
    samples = build("/source/kalfa/text_lines", path=str(root / "text.txt"))
    with pytest.raises(ValueError, match=re.escape(f"{name} {TABLE_ONLY}")):
        build(TRANSFORMS[name], **params)(samples)


def test_transform_set_applies_the_transforms_of_the_set_in_order():
    df = table()
    steps = [build("/transform/kalfa/filter", query="raw_0 > 1"),
             build("/transform/kalfa/derive", column="inter", expr="raw_0 * raw_1"),
             build("/transform/kalfa/drop", columns=["junk"])]
    out = build("/lego/kalfa/transform_set", df=df, set="train", transforms=steps)
    assert out.index.tolist() == [11, 12, 13]
    assert list(out.columns) == ["raw_0", "raw_1", "count", "inter"]
    assert out["inter"].tolist() == [40.0, 90.0, 160.0]
    expected = steps[2](steps[1](steps[0](df)))
    pandas.testing.assert_frame_equal(out, expected)
    reversed_out = build("/lego/kalfa/transform_set", df=df, set="valid", transforms=steps[::-1])
    assert list(reversed_out.columns) == ["raw_0", "raw_1", "count", "inter"]
    assert reversed_out.index.tolist() == [11, 12, 13]


def test_transform_set_without_transforms_passes_the_frame_untouched():
    df = table()
    assert build("/lego/kalfa/transform_set", df=df, set="train", transforms=[]) is df
    assert build("/lego/kalfa/transform_set", df=df, set="test", transforms=None) is df
    assert registry.facts("/lego/kalfa/transform_set").partial is False
