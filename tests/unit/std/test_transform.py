"""The transform legos and the per set step."""

import functools

import pandas

import kalfa  # noqa: F401
from kalfa.std.lego.kalfa.transform_set import transform_set
from kalfa.std.transform.kalfa.astype import astype
from kalfa.std.transform.kalfa.derive import derive
from kalfa.std.transform.kalfa.drop import drop
from kalfa.std.transform.kalfa.filter import filter_rows
from kalfa.std.transform.kalfa.rename import rename


def test_the_std_transforms_reshape_a_table():
    table = pandas.DataFrame({"cms_a_Z_score": [1.0, 2.0, 3.0], "b": [10, 20, 30]})
    renamed = rename(table, r"cms_(.*)_Z_score", r"z_\1")
    assert list(renamed.columns) == ["z_a", "b"]
    derived = derive(renamed, "tail", "z_a > 1.5")
    assert derived["tail"].tolist() == [False, True, True]
    kept = filter_rows(derived, "b < 30")
    assert kept["b"].tolist() == [10, 20]
    assert astype(kept, {"b": "float32"})["b"].dtype == "float32"
    assert list(drop(kept, ["tail"]).columns) == ["z_a", "b"]
    assert derive(table, "log_b", "log10(b)")["log_b"].round(2).tolist() == [1.0, 1.3, 1.48]


def test_transform_set_applies_the_named_steps_in_order():
    table = pandas.DataFrame({"a": [1, 2, 3, 4]})
    steps = [functools.partial(filter_rows, query="a > 1"), functools.partial(derive, column="b", expr="a * 2")]
    out = transform_set(table, "train", steps)
    assert out["a"].tolist() == [2, 3, 4] and out["b"].tolist() == [4, 6, 8]
    assert transform_set(table, "test", []) is table
