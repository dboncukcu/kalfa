"""simple_imputer, fill and the side outputs of a chain."""

import numpy
import pandas
import pytest

import kalfa  # noqa: F401
from kalfa.std.lego.kalfa.prep import apply, fit
from kalfa.std.pre.base import read_prep
from kalfa.std.pre.kalfa.missing import Fill, SimpleImputer
from kalfa.std.pre.sklearn.scalers import StandardScaler


def test_simple_imputer_fills_with_the_train_statistic_and_flags_the_gaps():
    values = numpy.array([1.0, numpy.nan, 3.0, numpy.nan])
    imputer = SimpleImputer(strategy="median", indicator=True)
    imputer.fit(values)
    assert imputer.statistic == 2.0 and imputer.apply(values).tolist() == [1.0, 2.0, 3.0, 2.0]
    assert imputer.extras(values)["missing"].tolist() == [False, True, False, True]
    assert SimpleImputer().extras(values) == {}
    words = numpy.array(["a", None, "a", "b"], dtype=object)
    frequent = SimpleImputer(strategy="most_frequent")
    frequent.fit(words)
    assert frequent.apply(words).tolist() == ["a", "a", "a", "b"]
    constant = SimpleImputer(strategy="constant", fill_value=-1.0)
    constant.fit(values)
    assert constant.apply(values).tolist() == [1.0, -1.0, 3.0, -1.0]
    assert constant.apply(numpy.array([1.0, 2.0])).tolist() == [1.0, 2.0]
    with pytest.raises(ValueError):
        SimpleImputer(strategy="mode")
    with pytest.raises(ValueError):
        SimpleImputer(strategy="constant")


def test_fill_needs_no_fit():
    assert Fill(value=0.0).apply(numpy.array([1.0, numpy.nan])).tolist() == [1.0, 0.0]
    assert Fill(method="ffill").apply(numpy.array([1.0, numpy.nan, 3.0, numpy.nan])).tolist() == [1.0, 1.0, 3.0, 3.0]
    assert Fill(method="bfill", value=9.0).apply(numpy.array([numpy.nan, 2.0, numpy.nan])).tolist() == [2.0, 2.0, 9.0]
    assert Fill(value="missing").apply(numpy.array(["a", None], dtype=object)).tolist() == ["a", "missing"]
    with pytest.raises(ValueError):
        Fill()
    with pytest.raises(ValueError):
        Fill(method="sideways")


def test_side_outputs_become_features_of_their_own(tmp_path):
    train = pandas.DataFrame({"a": [1.0, numpy.nan, 3.0, 4.0], "b": [1.0, 2.0, 3.0, 4.0],
                              "price": [1.0, 2.0, 3.0, 4.0]})
    templates = {"imputer": SimpleImputer(strategy="mean", indicator=True), "scaler": StandardScaler()}
    fields = {"a": {"preprocessors": ["imputer", "scaler"]}, "b": {"preprocessors": ["scaler"]},
              "price": {"target": True}}
    prep = fit(train, fields, templates, [], record=str(tmp_path))
    assert prep.features == ["a", "a_missing", "b"] and prep.field("a").extras == ["a_missing"]
    assert prep.dtypes["a_missing"] == "bool" and prep.field_of("a_missing") is None
    frame = apply(train, prep, "train")
    assert frame.data["a_missing"].tolist() == [False, True, False, False]
    matrix = frame.data[prep.features].to_numpy(dtype="float64")
    rescaled = prep.rescale_features(matrix)
    assert rescaled[:, 1].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert rescaled[:, 2].tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0])
    again = read_prep(tmp_path)
    assert again.features == prep.features and again.field("a").extras == ["a_missing"]
