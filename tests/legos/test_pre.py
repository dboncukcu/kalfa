import json
import math
import re

import numpy
import pandas
import pytest
import torch
from cirak.registry import registry
from PIL import Image

from helpers import build
from kalfa.std import STD_URIS
from kalfa.std.pre.base import (
    Affine,
    ColumnView,
    Encoder,
    Grouped,
    Prep,
    Preprocessor,
    SampleFrame,
    Scaler,
    StreamFrame,
    TableFrame,
    Tokenizer,
)
from kalfa.std.pre.kalfa.encoders import LabelEncoder, OneHot
from kalfa.std.pre.kalfa.images import RandomCropFlip, ToTensor
from kalfa.std.pre.kalfa.missing import SimpleImputer
from kalfa.std.pre.kalfa.scales import Absolute
from kalfa.std.pre.sklearn.scalers import StandardScaler


PRE = ["/pre/kalfa/abs", "/pre/kalfa/asinh", "/pre/kalfa/atanh", "/pre/kalfa/cast", "/pre/kalfa/char_tokenizer",
       "/pre/kalfa/fill", "/pre/kalfa/label_encoder", "/pre/kalfa/log", "/pre/kalfa/logit",
       "/pre/kalfa/median_std_scaler", "/pre/kalfa/normalize", "/pre/kalfa/one_hot", "/pre/kalfa/random_crop_flip",
       "/pre/kalfa/resize", "/pre/kalfa/simclr_aug", "/pre/kalfa/simple_imputer", "/pre/kalfa/sinh",
       "/pre/kalfa/tanh", "/pre/kalfa/to_tensor", "/pre/kalfa/to_tensor_signed", "/pre/kalfa/two_views",
       "/pre/sklearn/kbins_discretizer", "/pre/sklearn/max_abs_scaler", "/pre/sklearn/minmax_scaler",
       "/pre/sklearn/power_transformer", "/pre/sklearn/quantile_transformer", "/pre/sklearn/robust_scaler",
       "/pre/sklearn/spline_transformer", "/pre/sklearn/standard_scaler"]
PARAMS = {"/pre/kalfa/cast": {"dtype": "float32"}, "/pre/kalfa/fill": {"value": 0.0},
          "/pre/kalfa/normalize": {"mean": 0.5, "std": 0.5},
          "/pre/kalfa/random_crop_flip": {"size": 8}, "/pre/kalfa/resize": {"size": 4},
          "/pre/kalfa/simclr_aug": {"size": 8}, "/pre/kalfa/two_views": {"transform": None}}
GROUPED = {"/pre/sklearn/standard_scaler", "/pre/sklearn/minmax_scaler", "/pre/sklearn/max_abs_scaler",
           "/pre/sklearn/robust_scaler", "/pre/kalfa/median_std_scaler"}
STATEFUL = {"/pre/kalfa/char_tokenizer", "/pre/kalfa/one_hot", "/pre/kalfa/label_encoder"} | {
    uri for uri in PRE if uri.startswith("/pre/sklearn/")}
FIELDS = {"num_*": {"preprocessors": ["std"]}, "region": {"preprocessors": ["onehot"]},
          "late": {"preprocessors": ["impute", "std"]}, "tier": {"preprocessors": ["ordinal"]},
          "gap": {"preprocessors": ["abs_train", "std"]}, "y": {"target": True, "preprocessors": ["std"]}}
KEYS = {"abs_train": {"sets": ["train"]}, "std": {}}
FEATURES = ["num_0", "num_1", "region_east", "region_north", "region_south", "late", "late_missing", "tier", "gap"]
DTYPES = {"num_0": "float32", "num_1": "float32", "region_east": "float32", "region_north": "float32",
          "region_south": "float32", "late": "float32", "late_missing": "bool", "tier": "int64", "gap": "float32",
          "y": "float32"}
MEANS = {"num_0": 3.5, "num_1": 35.0, "late": 4.0, "gap": 2.0, "y": 7.0}
STDS = {"num_0": math.sqrt(17.5 / 6), "num_1": math.sqrt(1750.0 / 6), "late": math.sqrt(20.0 / 6),
        "gap": math.sqrt(4.0 / 6), "y": math.sqrt(70.0 / 6)}


def train_table():
    return pandas.DataFrame({"sample_id": [1, 2, 3, 4, 5, 6], "num_0": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                             "num_1": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                             "region": ["north", "south", "north", "east", "south", "north"],
                             "late": [1.0, numpy.nan, 3.0, numpy.nan, 5.0, 7.0],
                             "tier": ["low", "mid", "high", "low", "mid", "low"],
                             "gap": [-2.0, 1.0, -1.0, 2.0, -3.0, 3.0], "noise": [9, 9, 9, 9, 9, 9],
                             "y": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]}, index=[10, 11, 12, 13, 14, 15])


def held_table():
    return pandas.DataFrame({"sample_id": [7, 8, 9], "num_0": [3.5, 0.0, 7.0], "num_1": [35.0, 0.0, 70.0],
                             "region": ["south", "west", "north"], "late": [numpy.nan, 4.0, 2.0],
                             "tier": ["mid", "high", "low"], "gap": [-2.0, 4.0, 2.0], "noise": [9, 9, 9],
                             "y": [7.0, 0.0, 14.0]}, index=[20, 21, 22])


def templates():
    return {"std": build("/pre/sklearn/standard_scaler"), "onehot": build("/pre/kalfa/one_hot"),
            "impute": build("/pre/kalfa/simple_imputer", strategy="median", indicator=True),
            "ordinal": build("/pre/kalfa/label_encoder"), "abs_train": build("/pre/kalfa/abs")}


def fit_prep(record=None):
    return build("/lego/kalfa/fit", df=train_table(), fields=FIELDS, preprocessors=templates(), drop=["noise"],
                 keys=KEYS, record=record, spectators=["sample_id"])


def standardized(name, values):
    return (numpy.asarray(values, dtype="float64") - MEANS[name]) / STDS[name]


def gray_image(seed=0):
    values = (numpy.arange(64, dtype="int64").reshape(8, 8) * 4 + seed).astype("uint8")
    return Image.fromarray(values, mode="L")


def color_image():
    values = numpy.random.default_rng(3).integers(0, 256, size=(8, 8, 3), dtype="uint8")
    return Image.fromarray(values, mode="RGB")


def test_pre_catalog_is_the_documented_list():
    assert sorted(uri for uri in STD_URIS if uri.startswith("/pre/")) == PRE


@pytest.mark.parametrize("uri", PRE)
def test_pre_alias_is_the_name_and_the_facts_match_the_class(uri):
    assert registry.aliases()[uri.rsplit("/", 1)[1]] == uri
    facts = registry.facts(uri)
    assert facts.get("grouped") is (True if uri in GROUPED else None)
    assert facts.state is (uri in STATEFUL)
    params = PARAMS.get(uri, {})
    if uri == "/pre/kalfa/two_views":
        params = {"transform": build("/pre/kalfa/resize", size=4)}
    built = build(uri, **params)
    assert isinstance(built, Preprocessor)
    assert built.grouped is (uri in GROUPED)
    assert built.fits is (uri in STATEFUL or uri in ("/pre/kalfa/median_std_scaler", "/pre/kalfa/simple_imputer"))


def test_two_views_references_another_preprocessor():
    assert registry.facts("/pre/kalfa/two_views").refs == {"transform": "preprocessor"}


def test_standard_scaler_standardizes_every_column_of_its_block():
    scaler = build("/pre/sklearn/standard_scaler")
    assert isinstance(scaler, Affine)
    assert (scaler.grouped, scaler.fits, scaler.incremental, scaler.rescales) == (True, True, True, True)
    block = numpy.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [6.0, 40.0]])
    scaler.fit(block)
    shift, scale = scaler.affine()
    numpy.testing.assert_allclose(shift, [3.0, 25.0])
    numpy.testing.assert_allclose(scale, [math.sqrt(3.5), math.sqrt(125.0)])
    out = scaler.apply(block)
    assert out.shape == (4, 2)
    numpy.testing.assert_allclose(out, (block - [3.0, 25.0]) / [math.sqrt(3.5), math.sqrt(125.0)])
    numpy.testing.assert_allclose(scaler.inverse(out), block)
    column = scaler.apply(numpy.array([25.0, 35.0]), columns=[1])
    assert column.shape == (2,)
    numpy.testing.assert_allclose(column, [0.0, 10.0 / math.sqrt(125.0)])
    numpy.testing.assert_allclose(scaler.inverse(column, columns=[1]), [25.0, 35.0])
    with pytest.raises(ValueError, match=re.escape("the scaler was fitted on 2 columns and got 3")):
        scaler.apply(numpy.ones((2, 3)))


def test_standard_scaler_partial_fit_reaches_the_statistics_of_a_full_fit():
    block = numpy.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [6.0, 40.0]])
    full = build("/pre/sklearn/standard_scaler")
    full.fit(block)
    incremental = build("/pre/sklearn/standard_scaler")
    assert incremental.scaler is None
    incremental.partial_fit(block[:1])
    incremental.partial_fit(block[1:3])
    incremental.partial_fit(block[3:])
    assert int(numpy.max(incremental.scaler.n_samples_seen_)) == 4
    numpy.testing.assert_allclose(incremental.affine()[0], full.affine()[0])
    numpy.testing.assert_allclose(incremental.affine()[1], full.affine()[1])
    numpy.testing.assert_allclose(incremental.apply(block), full.apply(block))


def test_affine_scalers_invert_on_a_tensor_and_through_a_column_view():
    scaler = build("/pre/sklearn/standard_scaler")
    block = numpy.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [6.0, 40.0]])
    scaler.fit(block)
    scaled = torch.tensor(scaler.apply(block), dtype=torch.float32)
    back = scaler.inverse_torch(scaled)
    assert back.dtype == torch.float32
    numpy.testing.assert_allclose(back.numpy(), block, rtol=1e-6)
    assert "cached_terms" not in scaler.__getstate__()
    assert set(scaler.cached_terms) == {("cpu", torch.float32, None)}
    view = Grouped(scaler, ["a", "b"]).view("b")
    assert isinstance(view, ColumnView)
    assert view.position == 1
    assert view.rescales is True
    numpy.testing.assert_allclose(view.apply(numpy.array([25.0, 35.0])), [0.0, 10.0 / math.sqrt(125.0)])
    numpy.testing.assert_allclose(view.inverse(numpy.array([0.0, 1.0])), [25.0, 25.0 + math.sqrt(125.0)])
    numpy.testing.assert_allclose(view.inverse_torch(torch.tensor([0.0, 1.0])).numpy(),
                                  [25.0, 25.0 + math.sqrt(125.0)], rtol=1e-6)
    with pytest.raises(ValueError, match=re.escape("the scaler was fitted on 2 columns and got 3")):
        scaler.inverse_torch(torch.zeros(2, 3))
    one = build("/pre/sklearn/standard_scaler")
    one.fit(numpy.array([1.0, 3.0]))
    assert one.device_terms(torch.zeros(2)) == (2.0, 1.0)
    numpy.testing.assert_allclose(one.inverse_torch(torch.tensor([-1.0, 1.0])).numpy(), [1.0, 3.0])


def test_minmax_scaler_maps_the_train_range_onto_low_high():
    scaler = build("/pre/sklearn/minmax_scaler", low=-1.0, high=1.0)
    scaler.fit(numpy.array([0.0, 5.0, 10.0]))
    numpy.testing.assert_allclose(scaler.apply(numpy.array([0.0, 5.0, 10.0, 20.0])), [-1.0, 0.0, 1.0, 3.0])
    numpy.testing.assert_allclose(scaler.affine()[0], [5.0])
    numpy.testing.assert_allclose(scaler.affine()[1], [5.0])
    numpy.testing.assert_allclose(scaler.inverse(numpy.array([-1.0, 0.0, 1.0])), [0.0, 5.0, 10.0])
    unit = build("/pre/sklearn/minmax_scaler")
    assert (unit.low, unit.high) == (0.0, 1.0)
    unit.fit(numpy.array([[0.0, 100.0], [4.0, 300.0]]))
    numpy.testing.assert_allclose(unit.apply(numpy.array([[1.0, 200.0]])), [[0.25, 0.5]])
    numpy.testing.assert_allclose(unit.inverse_torch(torch.tensor([[0.25, 0.5]])).numpy(), [[1.0, 200.0]])


def test_max_abs_scaler_divides_by_the_largest_magnitude_keeping_sign_and_zeros():
    scaler = build("/pre/sklearn/max_abs_scaler")
    scaler.fit(numpy.array([-4.0, 2.0, 0.0, 1.0]))
    numpy.testing.assert_allclose(scaler.apply(numpy.array([-4.0, 2.0, 0.0, 1.0])), [-1.0, 0.5, 0.0, 0.25])
    numpy.testing.assert_allclose(scaler.affine()[0], [0.0])
    numpy.testing.assert_allclose(scaler.affine()[1], [4.0])
    numpy.testing.assert_allclose(scaler.inverse(numpy.array([-1.0, 0.25])), [-4.0, 1.0])


def test_robust_scaler_centers_on_the_median_and_scales_by_the_percentile_range():
    values = numpy.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0])
    scaler = build("/pre/sklearn/robust_scaler")
    assert (scaler.low, scaler.high) == (25.0, 75.0)
    scaler.fit(values)
    numpy.testing.assert_allclose(scaler.affine()[0], [5.5])
    numpy.testing.assert_allclose(scaler.affine()[1], [4.5])
    numpy.testing.assert_allclose(scaler.apply(numpy.array([5.5, 100.0, 1.0])), [0.0, 21.0, -1.0])
    numpy.testing.assert_allclose(scaler.inverse(numpy.array([0.0, 21.0])), [5.5, 100.0])
    wide = build("/pre/sklearn/robust_scaler", low=10.0, high=90.0)
    wide.fit(values)
    numpy.testing.assert_allclose(wide.affine()[1], [16.2])
    numpy.testing.assert_allclose(wide.apply(numpy.array([100.0])), [94.5 / 16.2])


def test_quantile_transformer_maps_a_column_onto_its_quantiles():
    values = numpy.arange(10, dtype="float64")
    transformer = build("/pre/sklearn/quantile_transformer", quantiles=10)
    assert isinstance(transformer, Scaler)
    assert transformer.output == "uniform"
    transformer.fit(values)
    assert transformer.transformer.n_quantiles_ == 10
    numpy.testing.assert_allclose(transformer.apply(values), values / 9.0)
    numpy.testing.assert_allclose(transformer.apply(numpy.array([4.5])), [0.5])
    numpy.testing.assert_allclose(transformer.inverse(values / 9.0), values)
    assert transformer.apply(values).shape == (10,)
    normal = build("/pre/sklearn/quantile_transformer", quantiles=11, output="normal", seed=3)
    assert normal.transformer is None
    normal.fit(numpy.arange(11, dtype="float64"))
    assert normal.transformer.random_state == 3
    out = normal.apply(numpy.arange(11, dtype="float64"))
    assert out[5] == 0.0
    numpy.testing.assert_allclose(out[:5], -out[6:][::-1])
    numpy.testing.assert_allclose(normal.inverse(out), numpy.arange(11, dtype="float64"), atol=1e-9)
    clipped = build("/pre/sklearn/quantile_transformer")
    clipped.fit(values)
    assert clipped.transformer.n_quantiles_ == 10
    floor = build("/pre/sklearn/quantile_transformer", quantiles=1)
    floor.fit(values)
    assert floor.transformer.n_quantiles_ == 2


def test_power_transformer_fits_the_exponent_and_standardizes():
    values = numpy.array([1.0, 2.0, 4.0, 8.0, 16.0])
    raw = build("/pre/sklearn/power_transformer", method="box-cox", standardize=False)
    raw.fit(values)
    assert raw.transformer.method == "box-cox"
    numpy.testing.assert_allclose(raw.transformer.lambdas_, [0.0], atol=1e-5)
    numpy.testing.assert_allclose(raw.apply(values), numpy.log(values), atol=1e-4)
    numpy.testing.assert_allclose(raw.inverse(raw.apply(values)), values, rtol=1e-6)
    standard = build("/pre/sklearn/power_transformer")
    assert (standard.method, standard.standardize) == ("yeo-johnson", True)
    standard.fit(values)
    out = standard.apply(values)
    numpy.testing.assert_allclose(out.mean(), 0.0, atol=1e-9)
    numpy.testing.assert_allclose(out.std(), 1.0, atol=1e-9)
    assert numpy.all(numpy.diff(out) > 0)
    numpy.testing.assert_allclose(standard.inverse(out), values, rtol=1e-6)
    signed = build("/pre/sklearn/power_transformer")
    signed.fit(numpy.array([-2.0, -1.0, 0.0, 1.0, 5.0]))
    assert signed.apply(numpy.array([-2.0, 5.0])).shape == (2,)
    with pytest.raises(ValueError, match="strictly positive"):
        raw.fit(numpy.array([-1.0, 1.0]))


def test_kbins_discretizer_writes_one_hot_bins_or_one_ordinal_column():
    values = numpy.arange(10, dtype="float64")
    onehot = build("/pre/sklearn/kbins_discretizer", bins=3, strategy="uniform")
    assert isinstance(onehot, Encoder)
    assert onehot.encode == "onehot"
    onehot.fit(values)
    assert onehot.width == 3
    assert onehot.columns("v") == ["v_bin0", "v_bin1", "v_bin2"]
    out = onehot.apply(values)
    assert out.dtype == numpy.float32
    assert out.shape == (10, 3)
    numpy.testing.assert_array_equal(out.argmax(axis=1), [0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    numpy.testing.assert_array_equal(out.sum(axis=1), numpy.ones(10))
    ordinal = build("/pre/sklearn/kbins_discretizer", bins=5, encode="ordinal")
    assert ordinal.strategy == "quantile"
    ordinal.fit(values)
    assert ordinal.width == 1
    assert ordinal.columns("v") == ["v_bin0"]
    codes = ordinal.apply(values)
    assert codes.dtype == numpy.float32
    assert codes.shape == (10,)
    numpy.testing.assert_array_equal(codes, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    kmeans = build("/pre/sklearn/kbins_discretizer", bins=3, strategy="kmeans", encode="ordinal")
    kmeans.fit(values)
    clusters = kmeans.apply(values)
    assert sorted(set(clusters.tolist())) == [0.0, 1.0, 2.0]
    assert numpy.all(numpy.diff(clusters) >= 0)


def test_spline_transformer_is_a_b_spline_basis_of_the_column():
    values = numpy.array([0.0, 2.5, 5.0, 7.5, 10.0])
    spline = build("/pre/sklearn/spline_transformer", knots=3, degree=1)
    assert isinstance(spline, Encoder)
    assert spline.extrapolation == "constant"
    spline.fit(values)
    assert spline.width == 2
    assert spline.columns("v") == ["v_spline0", "v_spline1"]
    out = spline.apply(values)
    assert out.dtype == numpy.float32
    numpy.testing.assert_array_equal(out, [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [0.0, 0.5], [0.0, 0.0]])
    numpy.testing.assert_array_equal(spline.apply(numpy.array([-5.0, 20.0])), [[1.0, 0.0], [0.0, 0.0]])
    cubic = build("/pre/sklearn/spline_transformer")
    assert (cubic.knots, cubic.degree) == (5, 3)
    cubic.fit(values)
    assert cubic.width == 6
    assert cubic.apply(values).shape == (5, 6)


def test_median_std_scaler_centers_on_the_median_and_scales_by_the_standard_deviation():
    scaler = build("/pre/kalfa/median_std_scaler")
    assert isinstance(scaler, Affine)
    assert (scaler.grouped, scaler.fits, scaler.incremental) == (True, True, False)
    block = numpy.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 5.0], [10.0, 5.0]])
    scaler.fit(block)
    numpy.testing.assert_allclose(scaler.center, [3.0, 5.0])
    numpy.testing.assert_allclose(scaler.scale, [math.sqrt(10.0), 1.0])
    out = scaler.apply(block)
    numpy.testing.assert_allclose(out[:, 0], (block[:, 0] - 3.0) / math.sqrt(10.0))
    numpy.testing.assert_allclose(out[:, 1], numpy.zeros(5))
    numpy.testing.assert_allclose(scaler.inverse(out), block)
    numpy.testing.assert_allclose(scaler.apply(numpy.array([4.0]), columns=[0]), [1.0 / math.sqrt(10.0)])
    numpy.testing.assert_allclose(scaler.inverse_torch(torch.tensor([[1.0, 2.0]])).numpy(),
                                  [[3.0 + math.sqrt(10.0), 7.0]], rtol=1e-6)
    with pytest.raises(ValueError, match=re.escape("the scaler was fitted on 2 columns and got 1")):
        scaler.apply(numpy.ones((3, 1)))
    gaps = build("/pre/kalfa/median_std_scaler")
    gaps.fit(numpy.array([1.0, numpy.nan, 3.0]))
    numpy.testing.assert_allclose(gaps.center, [2.0])
    numpy.testing.assert_allclose(gaps.scale, [1.0])
    out = gaps.apply(numpy.array([1.0, numpy.nan]))
    assert out[0] == -1.0 and numpy.isnan(out[1])


def test_simple_imputer_fills_missing_values_with_the_train_statistic():
    mean = build("/pre/kalfa/simple_imputer")
    assert isinstance(mean, SimpleImputer)
    assert (mean.strategy, mean.fits, mean.indicator) == ("mean", True, False)
    mean.fit(numpy.array([1.0, numpy.nan, 3.0]))
    assert mean.statistic == 2.0
    numpy.testing.assert_array_equal(mean.apply(numpy.array([numpy.nan, 5.0])), [2.0, 5.0])
    median = build("/pre/kalfa/simple_imputer", strategy="median")
    median.fit(numpy.array([1.0, numpy.nan, 4.0, 10.0]))
    assert median.statistic == 4.0
    frequent = build("/pre/kalfa/simple_imputer", strategy="most_frequent")
    frequent.fit(numpy.array([2.0, 1.0, numpy.nan, 1.0]))
    assert frequent.statistic == 1.0
    words = build("/pre/kalfa/simple_imputer", strategy="most_frequent")
    words.fit(numpy.array(["a", None, "b", "a"], dtype=object))
    assert words.statistic == "a"
    assert words.apply(numpy.array(["b", None], dtype=object)).tolist() == ["b", "a"]
    constant = build("/pre/kalfa/simple_imputer", strategy="constant", fill_value=7.0)
    constant.fit(numpy.array([numpy.nan]))
    assert constant.statistic == 7.0
    numpy.testing.assert_array_equal(constant.apply(numpy.array([numpy.nan, 1.0])), [7.0, 1.0])
    empty = build("/pre/kalfa/simple_imputer")
    empty.fit(numpy.array([numpy.nan, numpy.nan]))
    assert empty.statistic == 0.0
    complete = numpy.array([1.0, 2.0])
    assert mean.apply(complete).tolist() == [1.0, 2.0]
    assert mean.extras(numpy.array([numpy.nan, 1.0])) == {}


def test_simple_imputer_indicator_is_an_extra_computed_before_the_fill():
    imputer = build("/pre/kalfa/simple_imputer", strategy="median", indicator=True)
    imputer.fit(numpy.array([1.0, numpy.nan, 3.0]))
    extras = imputer.extras(numpy.array([numpy.nan, 1.0, numpy.nan]))
    assert list(extras) == ["missing"]
    assert extras["missing"].dtype == bool
    numpy.testing.assert_array_equal(extras["missing"], [True, False, True])
    numpy.testing.assert_array_equal(imputer.apply(numpy.array([numpy.nan, 1.0, numpy.nan])), [2.0, 1.0, 2.0])


def test_simple_imputer_validates_strategy_and_fill_value():
    with pytest.raises(ValueError, match=re.escape("simple_imputer.strategy must be one of ['mean', 'median', "
                                                   "'most_frequent', 'constant'], got 'mode'")):
        build("/pre/kalfa/simple_imputer", strategy="mode")
    with pytest.raises(ValueError, match=re.escape("simple_imputer.strategy constant needs fill_value")):
        build("/pre/kalfa/simple_imputer", strategy="constant")


def test_fill_fills_without_a_fit_by_value_or_along_the_rows():
    zero = build("/pre/kalfa/fill", value=0.0)
    assert zero.fits is False
    numpy.testing.assert_array_equal(zero.apply(numpy.array([1.0, numpy.nan, 3.0])), [1.0, 0.0, 3.0])
    gaps = numpy.array([numpy.nan, 1.0, numpy.nan, 2.0])
    numpy.testing.assert_array_equal(build("/pre/kalfa/fill", method="ffill").apply(gaps), [numpy.nan, 1.0, 1.0, 2.0])
    numpy.testing.assert_array_equal(build("/pre/kalfa/fill", method="bfill").apply(gaps), [1.0, 1.0, 2.0, 2.0])
    numpy.testing.assert_array_equal(build("/pre/kalfa/fill", method="ffill", value=9.0).apply(gaps),
                                     [9.0, 1.0, 1.0, 2.0])
    words = build("/pre/kalfa/fill", value="missing")
    assert words.apply(numpy.array(["a", None, "b"], dtype=object)).tolist() == ["a", "missing", "b"]
    with pytest.raises(ValueError, match=re.escape("fill.method must be ffill or bfill, got 'pad'")):
        build("/pre/kalfa/fill", method="pad")
    with pytest.raises(ValueError, match=re.escape("fill needs value or method")):
        build("/pre/kalfa/fill")


def test_cast_casts_a_column_to_the_numpy_dtype():
    cast = build("/pre/kalfa/cast", dtype="float32")
    assert cast.dtype == "float32"
    assert cast.fits is False
    out = cast.apply(numpy.array([1, 2]))
    assert out.dtype == numpy.float32
    assert out.tolist() == [1.0, 2.0]
    assert build("/pre/kalfa/cast", dtype="int64").apply(numpy.array([1.9, -0.5])).tolist() == [1, 0]
    assert build("/pre/kalfa/cast", dtype="bool").apply(numpy.array([0, 2])).tolist() == [False, True]


def test_one_hot_writes_a_column_per_sorted_category_and_zeros_for_unknown_ones():
    encoder = build("/pre/kalfa/one_hot")
    assert isinstance(encoder, Encoder)
    assert encoder.fits is True
    encoder.fit(numpy.array(["b", "a", "c", "a"], dtype=object))
    assert encoder.categories == ["a", "b", "c"]
    assert encoder.columns("region") == ["region_a", "region_b", "region_c"]
    out = encoder.apply(numpy.array(["c", "z", "a"], dtype=object))
    assert out.dtype == numpy.float32
    numpy.testing.assert_array_equal(out, [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    numbers = build("/pre/kalfa/one_hot")
    numbers.fit(numpy.array([3, 1, 3]))
    assert numbers.columns("k") == ["k_1", "k_3"]
    numpy.testing.assert_array_equal(numbers.apply(numpy.array([1])), [[1.0, 0.0]])


def test_label_encoder_codes_labels_in_sorted_order_and_decodes_scores():
    encoder = build("/pre/kalfa/label_encoder")
    assert isinstance(encoder, Encoder)
    assert encoder.decodes is True
    encoder.fit(numpy.array(["mid", "low", "high", "low"], dtype=object))
    assert encoder.classes.tolist() == ["high", "low", "mid"]
    codes = encoder.apply(numpy.array(["mid", "low", "high", "low"], dtype=object))
    assert codes.dtype == numpy.int64
    assert codes.tolist() == [2, 1, 0, 1]
    assert encoder.inverse(numpy.array([0, 2])).tolist() == ["high", "mid"]
    assert encoder.inverse(numpy.array([[1.0], [0.0]])).tolist() == ["low", "high"]
    assert encoder.decode(numpy.array([0.5, -0.5])).tolist() == ["low", "high"]
    assert encoder.decode(numpy.array([[0.5], [-0.5]])).tolist() == ["low", "high"]
    assert encoder.decode(numpy.array([[0.1, 0.7, 0.2], [0.9, 0.0, 0.1]])).tolist() == ["low", "high"]
    with pytest.raises(ValueError, match=re.escape("label 'zzz' was not seen when the encoder was fitted")):
        encoder.apply(numpy.array(["zzz"], dtype=object))
    numbers = build("/pre/kalfa/label_encoder")
    numbers.fit(numpy.array([3, 1, 2]))
    assert numbers.classes.tolist() == [1, 2, 3]
    assert numbers.apply(numpy.array([2, 3])).tolist() == [1, 2]


def test_char_tokenizer_builds_a_sorted_vocabulary_with_newline():
    tokenizer = build("/pre/kalfa/char_tokenizer")
    assert isinstance(tokenizer, Tokenizer)
    assert (tokenizer.dtype, tokenizer.fits) == ("int64", True)
    tokenizer.fit(numpy.array(["ab", "ba c"]))
    assert tokenizer.chars == ["\n", " ", "a", "b", "c"]
    assert tokenizer.size == 5
    encoded = tokenizer.encode("cab")
    assert encoded.dtype == numpy.int64
    assert encoded.tolist() == [4, 2, 3]
    assert tokenizer.encode("a\nz").tolist() == [2, 0, 1]
    assert tokenizer.decode([4, 2, 3]) == "cab"
    assert tokenizer.decode(numpy.array([[2], [0]])) == "a\n"
    assert tokenizer.apply("ba").tolist() == [3, 2]
    ids = numpy.array([1, 2], dtype="int64")
    assert tokenizer.apply(ids) is ids
    bare = build("/pre/kalfa/char_tokenizer")
    bare.fit([])
    assert bare.chars == ["\n"]
    assert bare.encode("q").tolist() == [0]


def test_abs_takes_the_absolute_value():
    absolute = build("/pre/kalfa/abs")
    assert isinstance(absolute, Absolute)
    assert (absolute.fits, absolute.rescales) == (False, False)
    assert absolute.apply(numpy.array([-1.5, 2.0, 0.0])).tolist() == [1.5, 2.0, 0.0]
    assert absolute.inverse(numpy.array([-1.0])).tolist() == [-1.0]


def test_log_is_log1p_over_norm_in_the_base():
    log = build("/pre/kalfa/log")
    assert (log.base, log.norm) == (10.0, 1.0)
    numpy.testing.assert_allclose(log.apply(numpy.array([9.0, 0.0, 99.0])), [1.0, 0.0, 2.0])
    numpy.testing.assert_allclose(log.inverse(numpy.array([1.0, 0.0, 2.0])), [9.0, 0.0, 99.0])
    numpy.testing.assert_allclose(log.inverse_torch(torch.tensor([1.0, 2.0])).numpy(), [9.0, 99.0], rtol=1e-5)
    natural = build("/pre/kalfa/log", base=math.e, norm=2.0)
    numpy.testing.assert_allclose(natural.apply(numpy.array([2.0 * (math.e - 1.0)])), [1.0])
    numpy.testing.assert_allclose(natural.inverse(numpy.array([1.0])), [2.0 * (math.e - 1.0)])


def test_asinh_compresses_the_tails_and_its_inverse_refuses_overflow():
    asinh = build("/pre/kalfa/asinh", scale=2.0)
    assert (asinh.scale, asinh.overflow) == (2.0, 700.0)
    numpy.testing.assert_allclose(asinh.apply(numpy.array([0.0, 2.0 * math.sinh(1.0), -2.0 * math.sinh(3.0)])),
                                  [0.0, 1.0, -3.0])
    numpy.testing.assert_allclose(asinh.inverse(numpy.array([1.0, -3.0])),
                                  [2.0 * math.sinh(1.0), -2.0 * math.sinh(3.0)])
    assert asinh.inverse(numpy.zeros(0)).shape == (0,)
    assert not hasattr(asinh, "inverse_torch")
    with pytest.raises(ValueError, match=re.escape("asinh: the inverse overflows at 800.0, past overflow=700.0")):
        asinh.inverse(numpy.array([1.0, -800.0]))
    with pytest.raises(ValueError, match=re.escape("asinh: scale must be positive, got 0")):
        build("/pre/kalfa/asinh", scale=0)
    tight = build("/pre/kalfa/asinh", overflow=5.0)
    with pytest.raises(ValueError, match="past overflow=5.0"):
        tight.inverse(numpy.array([6.0]))


def test_sinh_stretches_the_tails_and_refuses_overflow():
    sinh = build("/pre/kalfa/sinh", scale=2.0)
    numpy.testing.assert_allclose(sinh.apply(numpy.array([0.0, 2.0, -6.0])), [0.0, math.sinh(1.0), -math.sinh(3.0)])
    numpy.testing.assert_allclose(sinh.inverse(numpy.array([math.sinh(1.0), -math.sinh(3.0)])), [2.0, -6.0])
    numpy.testing.assert_allclose(sinh.inverse_torch(torch.tensor([math.sinh(1.0)])).numpy(), [2.0], rtol=1e-6)
    with pytest.raises(ValueError, match=re.escape("sinh: overflows at 1000.0 / scale, past overflow=700.0; raise "
                                                   "scale")):
        sinh.apply(numpy.array([2000.0]))
    with pytest.raises(ValueError, match=re.escape("sinh: scale must be positive, got -1.0")):
        build("/pre/kalfa/sinh", scale=-1.0)


def test_tanh_squashes_into_the_unit_interval_and_its_inverse_clips_at_the_edge():
    tanh = build("/pre/kalfa/tanh", scale=3.0)
    assert (tanh.scale, tanh.eps) == (3.0, 1e-15)
    numpy.testing.assert_allclose(tanh.apply(numpy.array([0.0, 3.0, -6.0])), [0.0, math.tanh(1.0), -math.tanh(2.0)])
    numpy.testing.assert_allclose(tanh.inverse(numpy.array([math.tanh(1.0), -math.tanh(2.0)])), [3.0, -6.0])
    edge = tanh.inverse(numpy.array([1.0, 5.0, -1.0]))
    assert numpy.isfinite(edge).all()
    assert edge[0] == edge[1] == -edge[2]
    numpy.testing.assert_allclose(edge[0], 3.0 * math.atanh(1.0 - 1e-15))
    assert tanh.apply(numpy.array([100.0]))[0] == 1.0
    assert not hasattr(tanh, "inverse_torch")
    with pytest.raises(ValueError, match=re.escape("tanh: scale must be positive, got 0.0")):
        build("/pre/kalfa/tanh", scale=0.0)


def test_atanh_opens_a_bounded_column_and_names_what_falls_outside():
    atanh = build("/pre/kalfa/atanh", scale=2.0)
    numpy.testing.assert_allclose(atanh.apply(numpy.array([0.0, 1.0, -1.0])), [0.0, math.atanh(0.5), -math.atanh(0.5)])
    numpy.testing.assert_allclose(atanh.inverse(numpy.array([math.atanh(0.5)])), [1.0])
    numpy.testing.assert_allclose(atanh.inverse_torch(torch.tensor([0.0, math.atanh(0.5)])).numpy(), [0.0, 1.0],
                                  rtol=1e-6)
    with pytest.raises(ValueError, match=re.escape("atanh takes values inside (-scale, scale); 2 of 3 are outside, "
                                                   "the largest is 3.0; raise scale")):
        atanh.apply(numpy.array([0.0, 3.0, -2.0]))
    with pytest.raises(ValueError, match=re.escape("atanh: scale must be positive, got -2")):
        build("/pre/kalfa/atanh", scale=-2)


def test_logit_clips_to_the_margins_and_inverts_by_the_sigmoid():
    logit = build("/pre/kalfa/logit", low=0.1, high=0.2)
    numpy.testing.assert_allclose(logit.apply(numpy.array([0.05, 0.5, 0.95])),
                                  [math.log(1.0 / 9.0), 0.0, math.log(4.0)])
    numpy.testing.assert_allclose(logit.inverse(numpy.array([0.0, math.log(4.0), math.log(1.0 / 9.0)])),
                                  [0.5, 0.8, 0.1])
    numpy.testing.assert_array_equal(logit.inverse(numpy.array([-1000.0, 1000.0])), [0.0, 1.0])
    numpy.testing.assert_allclose(logit.inverse_torch(torch.tensor([0.0, math.log(4.0)])).numpy(), [0.5, 0.8],
                                  rtol=1e-6)
    default = build("/pre/kalfa/logit")
    assert (default.low, default.high) == (1e-6, 1e-6)
    numpy.testing.assert_allclose(default.apply(numpy.array([0.0, 1.0])), [math.log(1e-6 / (1 - 1e-6)),
                                                                          math.log((1 - 1e-6) / 1e-6)])
    with pytest.raises(ValueError, match=re.escape("logit: low is a margin inside (0, 0.5), got 0.5")):
        build("/pre/kalfa/logit", low=0.5)
    with pytest.raises(ValueError, match=re.escape("logit: high is a margin inside (0, 0.5), got 0")):
        build("/pre/kalfa/logit", high=0)


def test_to_tensor_scales_an_image_into_the_unit_interval_channels_first():
    to_tensor = build("/pre/kalfa/to_tensor")
    assert isinstance(to_tensor, ToTensor)
    assert (to_tensor.dtype, to_tensor.signed, to_tensor.fits) == ("float32", False, False)
    image = gray_image()
    tensor = to_tensor.apply(image)
    assert tensor.shape == (1, 8, 8)
    assert tensor.dtype == torch.float32
    numpy.testing.assert_allclose(tensor[0].numpy(), numpy.asarray(image, dtype="float32") / 255.0)
    assert tensor.min().item() == 0.0
    assert tensor.max().item() == pytest.approx(252.0 / 255.0)
    color = to_tensor.apply(color_image())
    assert color.shape == (3, 8, 8)
    for channel in range(3):
        numpy.testing.assert_allclose(color[channel].numpy(), numpy.asarray(color_image())[:, :, channel] / 255.0)
    pair = to_tensor.apply([gray_image(), gray_image(1)])
    assert pair.shape == (2, 1, 8, 8)
    numpy.testing.assert_allclose(pair[1, 0].numpy(), numpy.asarray(gray_image(1), dtype="float32") / 255.0)
    with pytest.raises(TypeError, match=re.escape("expected an image, got int")):
        to_tensor.apply(5)


def test_to_tensor_signed_scales_into_the_signed_unit_interval():
    signed = build("/pre/kalfa/to_tensor_signed")
    assert signed.signed is True
    tensor = signed.apply(gray_image())
    assert tensor.shape == (1, 8, 8)
    numpy.testing.assert_allclose(tensor[0].numpy(), numpy.asarray(gray_image(), dtype="float32") / 255.0 * 2.0 - 1.0)
    assert tensor.min().item() == -1.0
    assert tensor.max().item() == pytest.approx(252.0 / 255.0 * 2.0 - 1.0)


def test_normalize_shifts_and_scales_per_channel_with_numbers_lists_or_presets():
    normalize = build("/pre/kalfa/normalize", mean=0.5, std=0.5)
    assert normalize.dtype == "float32"
    tensor = build("/pre/kalfa/to_tensor").apply(gray_image())
    out = normalize.apply(tensor)
    numpy.testing.assert_allclose(out.numpy(), (tensor.numpy() - 0.5) / 0.5, rtol=1e-6)
    color = build("/pre/kalfa/to_tensor").apply(color_image())
    lists = build("/pre/kalfa/normalize", mean=[0.1, 0.2, 0.3], std=[1.0, 2.0, 4.0])
    out = lists.apply(color)
    for channel, (mean, std) in enumerate([(0.1, 1.0), (0.2, 2.0), (0.3, 4.0)]):
        numpy.testing.assert_allclose(out[channel].numpy(), (color[channel].numpy() - mean) / std, rtol=1e-5)
    imagenet = build("/pre/kalfa/normalize", mean="imagenet", std="imagenet")
    assert imagenet.mean == [0.485, 0.456, 0.406]
    assert imagenet.std == [0.229, 0.224, 0.225]
    cifar = build("/pre/kalfa/normalize", mean="cifar10", std=1.0)
    assert cifar.mean == [0.4914, 0.4822, 0.4465]
    assert cifar.std == 1.0
    numpy.testing.assert_allclose(cifar.apply(color)[2].numpy(), color[2].numpy() - 0.4465, rtol=1e-5)
    with pytest.raises(KeyError):
        build("/pre/kalfa/normalize", mean="mnist", std=1.0)


def test_resize_takes_an_int_or_height_and_width():
    square = build("/pre/kalfa/resize", size=4)
    assert square.size == (4, 4)
    assert square.apply(gray_image()).size == (4, 4)
    assert numpy.asarray(square.apply(gray_image())).shape == (4, 4)
    tall = build("/pre/kalfa/resize", size=[2, 6])
    assert tall.size == (6, 2)
    assert tall.apply(gray_image()).size == (6, 2)
    assert numpy.asarray(tall.apply(gray_image())).shape == (2, 6)
    assert build("/pre/kalfa/resize", size=3.0).size == (3, 3)
    assert build("/pre/kalfa/resize", size=(5, 7)).apply(color_image()).size == (7, 5)


def test_random_crop_flip_crops_the_padded_image_and_repeats_under_the_seed():
    crop = build("/pre/kalfa/random_crop_flip", size=8)
    assert isinstance(crop, RandomCropFlip)
    assert (crop.size, crop.padding) == (8, 4)
    image = gray_image()
    torch.manual_seed(5)
    first = [build("/pre/kalfa/random_crop_flip", size=8).apply(image) for _ in range(3)]
    torch.manual_seed(5)
    second = [build("/pre/kalfa/random_crop_flip", size=8).apply(image) for _ in range(3)]
    assert [view.tobytes() for view in first] == [view.tobytes() for view in second]
    for view in first:
        assert view.size == (8, 8)
        assert view.mode == "L"
        assert set(numpy.asarray(view).reshape(-1).tolist()) <= set(numpy.asarray(image).reshape(-1).tolist()) | {0}
    small = build("/pre/kalfa/random_crop_flip", size=4).apply(image)
    assert small.size == (4, 4)


def test_simclr_aug_crops_resizes_and_jitters_under_the_seed():
    aug = build("/pre/kalfa/simclr_aug", size=6)
    assert (aug.size, aug.scale) == (6, (0.5, 1.0))
    assert build("/pre/kalfa/simclr_aug", size=6, scale=[0.2, 0.4]).scale == (0.2, 0.4)
    image = color_image()
    torch.manual_seed(9)
    first = [build("/pre/kalfa/simclr_aug", size=6).apply(image) for _ in range(3)]
    torch.manual_seed(9)
    second = [build("/pre/kalfa/simclr_aug", size=6).apply(image) for _ in range(3)]
    assert [view.tobytes() for view in first] == [view.tobytes() for view in second]
    for view in first:
        assert view.size == (6, 6)
        assert view.mode == "RGB"
    assert build("/pre/kalfa/simclr_aug", size=6).apply(gray_image()).mode == "L"


def test_two_views_applies_the_transform_twice_and_stacks_through_to_tensor():
    views = build("/pre/kalfa/two_views", transform=build("/pre/kalfa/resize", size=4))
    pair = views.apply(gray_image())
    assert isinstance(pair, tuple) and len(pair) == 2
    assert pair[0].size == (4, 4) and pair[1].size == (4, 4)
    assert pair[0].tobytes() == pair[1].tobytes()
    tensor = build("/pre/kalfa/to_tensor").apply(pair)
    assert tensor.shape == (2, 1, 4, 4)
    torch.manual_seed(1)
    random = build("/pre/kalfa/two_views", transform=build("/pre/kalfa/random_crop_flip", size=8))
    left, right = random.apply(gray_image())
    assert left.size == (8, 8) and right.size == (8, 8)
    assert build("/pre/kalfa/to_tensor_signed").apply((left, right)).shape == (2, 1, 8, 8)


def test_fit_resolves_the_field_globs_and_fits_every_chain_on_the_train_set(tmp_path):
    prep = fit_prep(str(tmp_path))
    assert isinstance(prep, Prep)
    assert [item.name for item in prep.fields] == ["num_0", "num_1", "region", "late", "tier", "gap", "y"]
    assert prep.features == FEATURES
    assert prep.targets == {"y": ["y"]}
    assert prep.dtypes == DTYPES
    assert prep.sets == {"abs_train": ["train"]}
    assert prep.drop == ["noise"]
    assert prep.spectators == ["sample_id"]
    assert [item.target for item in prep.fields] == [False, False, False, False, False, False, True]
    assert prep.field("region").columns == ["region_east", "region_north", "region_south"]
    assert prep.field("late").columns == ["late"]
    assert prep.field("late").extras == ["late_missing"]
    assert prep.field("gap").chain == ["abs_train", "std"]
    assert [item.extras for item in prep.fields if item.name != "late"] == [[]] * 6
    assert set(prep.fitted) == {"std", "onehot", "impute", "ordinal", "abs_train"}
    grouped = prep.fitted["std"]
    assert isinstance(grouped, Grouped)
    assert grouped.columns == ["num_0", "num_1", "late", "gap", "y"]
    assert isinstance(grouped.preprocessor, StandardScaler)
    numpy.testing.assert_allclose(grouped.preprocessor.affine()[0], [MEANS[name] for name in grouped.columns])
    numpy.testing.assert_allclose(grouped.preprocessor.affine()[1], [STDS[name] for name in grouped.columns])
    assert list(prep.fitted["onehot"]) == ["region"]
    assert isinstance(prep.fitted["onehot"]["region"], OneHot)
    assert prep.fitted["onehot"]["region"].categories == ["east", "north", "south"]
    assert prep.fitted["impute"]["late"].statistic == 4.0
    assert prep.fitted["ordinal"]["tier"].classes.tolist() == ["high", "low", "mid"]
    assert isinstance(prep.fitted["abs_train"]["gap"], Absolute)
    assert isinstance(prep.object_of("std", "y"), ColumnView)
    assert prep.object_of("std", "y").position == 4
    assert prep.object_of("std", "region") is None
    assert prep.object_of("onehot", "region") is prep.fitted["onehot"]["region"]


def test_fit_writes_the_plan_and_the_pickles_that_read_prep_reads_back(tmp_path):
    prep = fit_prep(str(tmp_path))
    folder = tmp_path / "fitted" / "preprocessors"
    assert sorted(path.name for path in folder.iterdir()) == ["abs_train.pkl", "impute.pkl", "onehot.pkl",
                                                               "ordinal.pkl", "plan.json", "std.pkl"]
    plan = json.loads((folder / "plan.json").read_text())
    assert plan == prep.plan()
    assert list(plan) == ["fields", "sets", "dtypes", "drop", "spectators"]
    assert plan["fields"][3] == {"name": "late", "chain": ["impute", "std"], "target": False, "columns": ["late"],
                                 "extras": ["late_missing"]}
    assert plan["fields"][6] == {"name": "y", "chain": ["std"], "target": True, "columns": ["y"], "extras": []}
    read = build("/lego/kalfa/read_prep", record=str(tmp_path))
    assert isinstance(read, Prep)
    assert read.plan() == prep.plan()
    assert read.features == FEATURES
    assert isinstance(read.fitted["std"], Grouped)
    assert read.fitted["std"].columns == ["num_0", "num_1", "late", "gap", "y"]
    numpy.testing.assert_allclose(read.fitted["std"].preprocessor.affine()[0],
                                  prep.fitted["std"].preprocessor.affine()[0])
    assert read.fitted["ordinal"]["tier"].classes.tolist() == ["high", "low", "mid"]
    assert read.fitted["impute"]["late"].statistic == 4.0
    assert read.sets == {"abs_train": ["train"]}
    assert not (tmp_path / "other").exists()
    assert fit_prep() is not None
    assert registry.facts("/lego/kalfa/fit").returns == "prep"
    assert registry.facts("/lego/kalfa/fit").state is True
    assert registry.facts("/lego/kalfa/fit").bus == {"record": "record"}
    assert registry.facts("/lego/kalfa/read_prep").returns == "prep"


def test_apply_applies_the_fitted_chains_to_a_set_with_its_sets_mask_and_spectators():
    prep = fit_prep()
    frame = build("/lego/kalfa/apply", df=held_table(), prep=prep, set="test", keys=KEYS, mask="num_0 > 5")
    assert isinstance(frame, TableFrame)
    assert (frame.set, frame.features, frame.targets) == ("test", FEATURES, {"y": ["y"]})
    assert list(frame.data.columns) == FEATURES + ["y"]
    assert frame.data.index.tolist() == [20, 21, 22]
    assert frame.index.tolist() == [20, 21, 22]
    assert len(frame) == 3
    assert {name: str(dtype) for name, dtype in frame.data.dtypes.items()} == DTYPES
    data = frame.data
    numpy.testing.assert_allclose(data["num_0"], standardized("num_0", [3.5, 0.0, 7.0]), rtol=1e-6)
    numpy.testing.assert_allclose(data["num_1"], standardized("num_1", [35.0, 0.0, 70.0]), rtol=1e-6)
    numpy.testing.assert_array_equal(data[["region_east", "region_north", "region_south"]].to_numpy(),
                                     [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    numpy.testing.assert_allclose(data["late"], standardized("late", [4.0, 4.0, 2.0]), rtol=1e-6)
    assert data["late_missing"].tolist() == [True, False, False]
    assert data["tier"].tolist() == [2, 0, 1]
    numpy.testing.assert_allclose(data["gap"], standardized("gap", [-2.0, 4.0, 2.0]), rtol=1e-6)
    numpy.testing.assert_allclose(data["y"], standardized("y", [7.0, 0.0, 14.0]), rtol=1e-6)
    numpy.testing.assert_array_equal(frame.mask, [True, True, False])
    assert frame.mask.dtype == bool
    assert list(frame.extra.columns) == ["sample_id"]
    assert frame.extra["sample_id"].tolist() == [7, 8, 9]
    assert frame.extra.index.tolist() == [20, 21, 22]


def test_apply_on_the_train_set_runs_the_preprocessors_limited_to_it():
    prep = fit_prep()
    train = build("/lego/kalfa/apply", df=held_table(), prep=prep, set="train", keys=KEYS)
    numpy.testing.assert_allclose(train.data["gap"], standardized("gap", [2.0, 4.0, 2.0]), rtol=1e-6)
    assert train.mask is None
    assert train.set == "train"
    own = build("/lego/kalfa/apply", df=held_table(), prep=prep, set="valid")
    numpy.testing.assert_allclose(own.data["gap"], standardized("gap", [-2.0, 4.0, 2.0]), rtol=1e-6)
    everywhere = build("/lego/kalfa/apply", df=held_table(), prep=prep, set="valid", keys={})
    numpy.testing.assert_allclose(everywhere.data["gap"], standardized("gap", [2.0, 4.0, 2.0]), rtol=1e-6)
    fitted = build("/lego/kalfa/apply", df=train_table(), prep=prep, set="train", keys=KEYS)
    numpy.testing.assert_allclose(fitted.data[["num_0", "num_1", "late", "gap", "y"]].to_numpy().mean(axis=0),
                                  numpy.zeros(5), atol=1e-6)
    numpy.testing.assert_allclose(fitted.data[["num_0", "num_1", "late", "gap", "y"]].to_numpy().std(axis=0),
                                  numpy.ones(5), rtol=1e-6)
    assert fitted.data["late_missing"].tolist() == [False, True, False, True, False, False]
    assert list(fitted.extra.columns) == ["sample_id"]


def test_apply_types_an_empty_set_by_the_plan_and_names_a_missing_column():
    fields = {name: spec for name, spec in FIELDS.items() if name != "region"}
    prep = build("/lego/kalfa/fit", df=train_table(), fields=fields, preprocessors=templates(),
                 drop=["noise", "region"], keys=KEYS)
    dtypes = {name: kind for name, kind in DTYPES.items() if not name.startswith("region_")}
    assert prep.dtypes == dtypes
    empty = build("/lego/kalfa/apply", df=held_table().iloc[:0], prep=prep, set="test", keys=KEYS, mask="num_0 > 5")
    assert len(empty) == 0
    assert list(empty.data.columns) == list(dtypes)
    assert {name: str(dtype) for name, dtype in empty.data.dtypes.items()} == dtypes
    assert empty.mask.shape == (0,)
    assert list(empty.extra.columns) == []
    with pytest.raises(ValueError, match=re.escape("the test data lacks column 'late'")):
        build("/lego/kalfa/apply", df=held_table().drop(columns=["late"]), prep=prep, set="test", keys=KEYS)


def test_apply_types_an_empty_set_with_a_one_hot_field():
    prep = fit_prep()
    empty = build("/lego/kalfa/apply", df=held_table().iloc[:0], prep=prep, set="test", keys=KEYS)
    assert len(empty) == 0
    assert list(empty.data.columns) == list(DTYPES)
    assert {name: str(dtype) for name, dtype in empty.data.dtypes.items()} == DTYPES


def test_prep_inverts_rescales_and_decodes_through_the_chains():
    prep = fit_prep()
    numpy.testing.assert_allclose(prep.inverse("y", [0.0, 1.0]), [7.0, 7.0 + STDS["y"]])
    numpy.testing.assert_allclose(prep.rescale("y", numpy.array([0.0, 1.0])), [7.0, 7.0 + STDS["y"]])
    numpy.testing.assert_allclose(prep.rescale_torch("y", torch.tensor([0.0, 1.0])).numpy(),
                                  [7.0, 7.0 + STDS["y"]], rtol=1e-6)
    numpy.testing.assert_allclose(prep.inverse("gap", [0.0], set_name="test"), [2.0])
    numpy.testing.assert_allclose(prep.inverse("gap", [0.0], set_name="train"), [2.0])
    assert prep.rescales() is True
    assert prep.rescales("y") is True
    assert prep.rescales("tier") is False
    assert prep.rescales("region") is False
    assert prep.rescales_on_device(["y", "num_0"]) is True
    assert isinstance(prep.decoder("tier"), LabelEncoder)
    assert prep.decoder("y") is None
    assert prep.decode("tier", numpy.array([[0.1, 0.8, 0.1], [0.9, 0.0, 0.1]])).tolist() == ["low", "high"]
    assert prep.decode("y", numpy.array([0.5])) is None
    assert prep.tokenizer() is None
    assert (prep.applies("abs_train", "train"), prep.applies("abs_train", "test"), prep.applies("std", "test")) == (
        True, False, True)
    assert prep.field_of("region_north") == "region"
    assert prep.field_of("late_missing") is None
    assert prep.field_of("zzz") is None
    rescaled = prep.rescale_features(numpy.zeros((2, 9)))
    numpy.testing.assert_allclose(rescaled, [[3.5, 35.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 2.0]] * 2)
    numpy.testing.assert_array_equal(prep.rescale_features(numpy.zeros((2, 3))), numpy.zeros((2, 3)))


def test_fit_rejects_unknown_preprocessors_unmatched_fields_and_reserved_names():
    with pytest.raises(ValueError, match=re.escape("fields name preprocessors ['nope'] that data.preprocessors does "
                                                   "not define")):
        build("/lego/kalfa/fit", df=train_table(), fields={"num_0": {"preprocessors": ["nope"]}},
              preprocessors=templates(), drop=[])
    with pytest.raises(ValueError, match=re.escape("field 'zzz' matches no column; the columns are ['sample_id', "
                                                   "'num_0', 'num_1', 'region', 'late', 'tier', 'gap', 'y']")):
        build("/lego/kalfa/fit", df=train_table(), fields={"zzz": {}}, preprocessors={}, drop=["noise"])
    with pytest.raises(ValueError, match=re.escape("column 'num_0' matches 'n*' and '*0' with equal specificity")):
        build("/lego/kalfa/fit", df=train_table(), fields={"n*": {"preprocessors": ["std"]},
                                                           "*0": {"preprocessors": ["std"]}},
              preprocessors=templates(), drop=["noise"])
    with pytest.raises(ValueError, match=re.escape("'x' is reserved for the feature tensor; rename the column")):
        build("/lego/kalfa/fit", df=train_table().rename(columns={"num_0": "x"}), fields={"x": {}},
              preprocessors={}, drop=[])
    with pytest.raises(ValueError, match=re.escape("'input' is a reserved field name")):
        build("/lego/kalfa/fit", df=train_table().rename(columns={"num_0": "input"}), fields={"input": {}},
              preprocessors={}, drop=[])
    target = build("/lego/kalfa/fit", df=train_table().rename(columns={"y": "x"}),
                   fields={"x": {"target": True}}, preprocessors={}, drop=[])
    assert target.targets == {"x": ["x"]}


def test_fit_prefers_a_literal_field_and_then_the_more_specific_glob():
    scalers = {"std": build("/pre/sklearn/standard_scaler"), "mm": build("/pre/sklearn/minmax_scaler"),
               "ma": build("/pre/sklearn/max_abs_scaler")}
    prep = build("/lego/kalfa/fit", df=train_table(),
                 fields={"n*": {"preprocessors": ["std"]}, "num_*": {"preprocessors": ["mm"]},
                         "num_1": {"preprocessors": ["ma"]}}, preprocessors=scalers, drop=["noise"])
    assert prep.field("num_0").chain == ["mm"]
    assert prep.field("num_1").chain == ["ma"]
    assert [item.name for item in prep.fields] == ["num_0", "num_1"]
    assert set(prep.fitted) == {"mm", "ma"}


def test_fit_refuses_a_wide_column_in_a_grouped_scaler_and_chains_in_different_orders():
    with pytest.raises(ValueError, match=re.escape("preprocessor 'std' fits over all its columns at once, so every "
                                                   "column reaching it must be one column wide; ['region'] are "
                                                   "wider")):
        build("/lego/kalfa/fit", df=train_table(), fields={"region": {"preprocessors": ["onehot", "std"]}},
              preprocessors=templates(), drop=["noise"])
    scalers = {"std": build("/pre/sklearn/standard_scaler"), "mm": build("/pre/sklearn/minmax_scaler")}
    with pytest.raises(ValueError, match=re.escape("the chains of ['num_0', 'num_1'] cannot be fitted: "
                                                   "preprocessors that fit over all their columns are written in "
                                                   "different orders")):
        build("/lego/kalfa/fit", df=train_table(), fields={"num_0": {"preprocessors": ["std", "mm"]},
                                                           "num_1": {"preprocessors": ["mm", "std"]}},
              preprocessors=scalers, drop=["noise"])
    with pytest.raises(TypeError, match=re.escape("field 'region' has dtype object after its chain; add a "
                                                  "preprocessor that turns it into a number (cast, one_hot, "
                                                  "label_encoder, a tokenizer)")):
        build("/lego/kalfa/fit", df=train_table(), fields={"region": {}}, preprocessors={}, drop=["noise"])


def test_fit_on_a_stream_fits_per_column_chunk_by_chunk(tmp_path):
    train_table().to_parquet(tmp_path / "train.parquet", index=False)
    stream = build("/source/kalfa/parquet_stream", path=str(tmp_path / "train.parquet"), chunk=2)
    fields = {"num_*": {"preprocessors": ["std"]}, "late": {"preprocessors": ["impute", "std"]},
              "gap": {"preprocessors": ["abs_train", "std"]}, "y": {"target": True, "preprocessors": ["std"]}}
    prep = build("/lego/kalfa/fit", df=stream, fields=fields, preprocessors=templates(), drop=["noise"], keys=KEYS,
                 record=str(tmp_path / "record"))
    assert prep.features == ["num_0", "num_1", "late", "late_missing", "gap"]
    assert prep.targets == {"y": ["y"]}
    assert isinstance(prep.fitted["std"], dict)
    assert list(prep.fitted["std"]) == ["num_0", "num_1", "late", "gap", "y"]
    for name, scaler in prep.fitted["std"].items():
        numpy.testing.assert_allclose(scaler.affine()[0], [MEANS[name]])
        numpy.testing.assert_allclose(scaler.affine()[1], [STDS[name]])
        assert int(numpy.max(scaler.scaler.n_samples_seen_)) == 6
    assert prep.fitted["impute"]["late"].statistic == 4.0
    assert prep.dtypes == {"num_0": "float32", "num_1": "float32", "late": "float32", "late_missing": "bool",
                           "gap": "float32", "y": "float32"}
    frame = build("/lego/kalfa/apply", df=stream, prep=prep, set="train", keys=KEYS)
    assert isinstance(frame, StreamFrame)
    assert frame.features == prep.features
    with pytest.raises(TypeError, match="a stream frame has no length; it is read in chunks"):
        len(frame)
    chunks = list(frame.stream.chunks())
    assert [len(index) for index, _ in chunks] == [2, 2, 2]
    numpy.testing.assert_array_equal(numpy.concatenate([index for index, _ in chunks]), numpy.arange(6))
    data = pandas.concat([part for _, part in chunks])
    assert list(data.columns) == prep.features + ["y"]
    numpy.testing.assert_allclose(data["num_0"], standardized("num_0", [1, 2, 3, 4, 5, 6]), rtol=1e-6)
    numpy.testing.assert_allclose(data["gap"], standardized("gap", [2, 1, 1, 2, 3, 3]), rtol=1e-6)
    assert data["late_missing"].tolist() == [False, True, False, True, False, False]
    read = build("/lego/kalfa/read_prep", record=str(tmp_path / "record"))
    assert read.plan() == prep.plan()


def test_fit_on_a_stream_skips_a_preprocessor_not_fitted_on_train_and_apply_names_missing_columns(tmp_path):
    train_table().to_parquet(tmp_path / "train.parquet", index=False)
    stream = build("/source/kalfa/parquet_stream", path=str(tmp_path / "train.parquet"), chunk=4)
    prep = build("/lego/kalfa/fit", df=stream, fields={"gap": {"preprocessors": ["abs_train", "std"]}},
                 preprocessors=templates(), drop=[], keys={"abs_train": {"sets": ["test"]}})
    assert set(prep.fitted) == {"std"}
    numpy.testing.assert_allclose(prep.fitted["std"]["gap"].affine()[0], [0.0])
    other = build("/source/kalfa/parquet_stream", path=str(tmp_path / "train.parquet"), chunk=4,
                  columns=["num_0", "y"])
    with pytest.raises(ValueError, match=re.escape("the test data lacks columns ['gap']")):
        build("/lego/kalfa/apply", df=other, prep=prep, set="test")


def test_fit_on_a_dataset_source_fits_on_the_readable_fields_and_types_by_the_chain(root):
    lines = build("/source/kalfa/text_lines", path=str(root / "text.txt"))
    prep = build("/lego/kalfa/fit", df=lines, fields={"text": {"preprocessors": ["tok"]}},
                 preprocessors={"tok": build("/pre/kalfa/char_tokenizer")}, drop=[])
    assert [item.name for item in prep.fields] == ["text"]
    assert prep.field("text").columns == ["text"]
    assert prep.features == ["text"]
    assert prep.dtypes == {"text": "int64"}
    tokenizer = prep.tokenizer()
    assert isinstance(tokenizer, Tokenizer)
    text = (root / "text.txt").read_text(encoding="utf-8")
    assert tokenizer.chars == sorted(set(text) | {"\n"})
    assert tokenizer.size == len(set(text) | {"\n"})
    images = build("/source/kalfa/image_folder", path=str(root / "images"))
    prep = build("/lego/kalfa/fit", df=images, fields={"image": {"preprocessors": ["to_t"]}, "label": {"target": True}},
                 preprocessors={"to_t": build("/pre/kalfa/to_tensor")}, drop=[])
    assert prep.features == ["image"]
    assert prep.targets == {"label": ["label"]}
    assert prep.dtypes == {"image": "float32", "label": "int64"}
    assert isinstance(prep.fitted["to_t"]["image"], ToTensor)


def test_fit_on_a_dataset_source_rejects_globs_unreadable_fields_and_non_tensor_chains(root):
    images = build("/source/kalfa/image_folder", path=str(root / "images"))
    with pytest.raises(ValueError, match=re.escape("pattern fields ['im*'] need a table; a Dataset source yields the "
                                                   "fields it declares, list them by name")):
        build("/lego/kalfa/fit", df=images, fields={"im*": {}}, preprocessors={}, drop=[])
    with pytest.raises(ValueError, match=re.escape("preprocessor 'std' fits on a column, but field 'image' of the "
                                                   "Dataset source cannot be read as one")):
        build("/lego/kalfa/fit", df=images, fields={"image": {"preprocessors": ["std"]}},
              preprocessors={"std": build("/pre/sklearn/standard_scaler")}, drop=[])
    with pytest.raises(TypeError, match=re.escape("field 'image' has dtype image after its chain; add a preprocessor "
                                                  "that turns it into a tensor (to_tensor)")):
        build("/lego/kalfa/fit", df=images, fields={"image": {}}, preprocessors={}, drop=[])


def test_apply_on_a_dataset_source_is_a_sample_frame_with_the_chains_of_the_set(root):
    images = build("/source/kalfa/image_folder", path=str(root / "images"))
    preprocessors = {"crop": build("/pre/kalfa/random_crop_flip", size=8), "to_t": build("/pre/kalfa/to_tensor")}
    keys = {"crop": {"sets": ["train"]}}
    prep = build("/lego/kalfa/fit", df=images, fields={"image": {"preprocessors": ["crop", "to_t"]},
                                                       "label": {"target": True}},
                 preprocessors=preprocessors, drop=[], keys=keys)
    test = build("/lego/kalfa/apply", df=images.subset([0, 40]), prep=prep, set="test", keys=keys)
    assert isinstance(test, SampleFrame)
    assert (test.set, test.features, test.targets) == ("test", [], {"label": ["label"]})
    assert test.fields == ["image", "label"]
    assert [type(item) for item in test.chains["image"]] == [ToTensor]
    assert test.chains["label"] == []
    assert len(test) == 2
    numpy.testing.assert_array_equal(test.index, [0, 40])
    train = build("/lego/kalfa/apply", df=images, prep=prep, set="train", keys=keys)
    assert [type(item) for item in train.chains["image"]] == [RandomCropFlip, ToTensor]
    assert len(train) == 64
    lines = build("/source/kalfa/text_lines", path=str(root / "text.txt"))
    with pytest.raises(ValueError, match=re.escape("the test data lacks field 'image'")):
        build("/lego/kalfa/apply", df=lines, prep=prep, set="test", keys=keys)
