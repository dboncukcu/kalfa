import re

import numpy
import pandas
import pytest
from cirak.registry import registry

from helpers import build
from kalfa.std import STD_URIS
from kalfa.std.frame.base import FrameTransform
from kalfa.std.frame.kalfa.statistics import GroupStatistic, TargetEncoding


def train_table():
    return pandas.DataFrame({"site": ["s0", "s0", "s1", "s0", "s1"], "tier": ["low", "high", "low", "low", "high"],
                             "v": [1.0, 2.0, 4.0, 6.0, 10.0], "y": [1.0, 2.0, 4.0, 6.0, 10.0]},
                            index=[10, 11, 12, 13, 14])


def held_table():
    return pandas.DataFrame({"site": ["s1", "s9", "s0"], "tier": ["high", "low", "low"], "v": [0.0, 0.0, 0.0],
                             "y": [0.0, 0.0, 0.0]}, index=[7, 8, 9])


def test_frame_catalog_is_the_two_statistics():
    assert sorted(uri for uri in STD_URIS if uri.startswith("/frame/")) == ["/frame/kalfa/group_statistic",
                                                                            "/frame/kalfa/target_encoding"]


def test_frame_aliases_and_facts():
    aliases = registry.aliases()
    assert aliases["group_statistic"] == "/frame/kalfa/group_statistic"
    assert aliases["target_encoding"] == "/frame/kalfa/target_encoding"
    assert registry.facts("/frame/kalfa/group_statistic").refs == {"by": "column"}
    assert registry.facts("/frame/kalfa/target_encoding").refs == {"column": "column"}
    assert registry.facts("/frame/kalfa/group_statistic").get("needs_table") is True
    assert registry.facts("/frame/kalfa/target_encoding").get("needs_table") is True
    assert registry.facts("/lego/kalfa/fit_frames").returns == "frames"
    assert registry.facts("/lego/kalfa/fit_frames").state is True
    assert registry.facts("/lego/kalfa/fit_frames").bus == {"record": "record"}
    assert registry.facts("/lego/kalfa/read_frames").returns == "frames"


def test_group_statistic_maps_the_train_statistic_of_every_group_onto_a_set():
    transform = build("/frame/kalfa/group_statistic", by="site", column="v", statistic="median")
    assert isinstance(transform, FrameTransform)
    assert transform.name == "v_median_by_site"
    assert transform.by == ["site"]
    train = train_table()
    transform.fit(train)
    assert transform.table.to_dict() == {"s0": 2.0, "s1": 7.0}
    assert transform.overall == 4.0
    out = transform.apply(held_table())
    assert list(out.columns) == ["site", "tier", "v", "y", "v_median_by_site"]
    assert out.index.tolist() == [7, 8, 9]
    assert out["v_median_by_site"].tolist() == [7.0, 4.0, 2.0]
    assert str(out["v_median_by_site"].dtype) == "float64"
    fitted = transform.apply(train)
    assert fitted["v_median_by_site"].tolist() == [2.0, 2.0, 7.0, 2.0, 7.0]
    assert list(train.columns) == ["site", "tier", "v", "y"]


def test_group_statistic_takes_a_name_and_every_statistic():
    train = train_table()
    expected = {"mean": ([3.0, 7.0], 4.6), "min": ([1.0, 4.0], 1.0), "max": ([6.0, 10.0], 10.0),
                "count": ([3.0, 2.0], 5.0)}
    for statistic, (per_group, overall) in expected.items():
        transform = build("/frame/kalfa/group_statistic", by="site", column="v", statistic=statistic, name="feat")
        transform.fit(train)
        assert transform.name == "feat"
        assert transform.table.astype("float64").tolist() == per_group
        assert transform.overall == overall
        assert transform.apply(held_table())["feat"].tolist() == [per_group[1], overall, per_group[0]]
    spread = build("/frame/kalfa/group_statistic", by="site", column="v", statistic="std")
    spread.fit(train)
    numpy.testing.assert_allclose(spread.table.to_numpy(), [numpy.sqrt(7.0), numpy.sqrt(18.0)])
    numpy.testing.assert_allclose(spread.overall, train["v"].std())
    assert build("/frame/kalfa/group_statistic", by="site", column="v").statistic == "mean"


def test_group_statistic_over_several_columns_keys_on_their_tuples():
    transform = build("/frame/kalfa/group_statistic", by=["site", "tier"], column="v")
    assert transform.name == "v_mean_by_site_tier"
    transform.fit(train_table())
    assert transform.table.to_dict() == {("s0", "low"): 3.5, ("s0", "high"): 2.0, ("s1", "low"): 4.0,
                                         ("s1", "high"): 10.0}
    out = transform.apply(held_table())
    assert out["v_mean_by_site_tier"].tolist() == [10.0, 4.6, 3.5]


def test_group_statistic_validates_the_statistic_and_needs_a_table(root):
    with pytest.raises(ValueError, match=re.escape("group_statistic.statistic must be one of ['mean', 'median', "
                                                   "'min', 'max', 'std', 'count'], got 'sum'")):
        build("/frame/kalfa/group_statistic", by="site", column="v", statistic="sum")
    transform = build("/frame/kalfa/group_statistic", by="site", column="v")
    stream = build("/source/kalfa/parquet_stream", path=str(root / "housing.parquet"))
    message = "group_statistic fits on a table in memory; a stream or a Dataset source has no frame to fit on"
    with pytest.raises(ValueError, match=re.escape(message)):
        transform.fit(stream)
    transform.fit(train_table())
    with pytest.raises(ValueError, match=re.escape(message)):
        transform.apply(build("/source/kalfa/text_lines", path=str(root / "text.txt")))


def test_target_encoding_smooths_the_category_means_toward_the_overall_mean():
    transform = build("/frame/kalfa/target_encoding", column="site", target="y", smoothing=2.0)
    assert isinstance(transform, FrameTransform)
    assert transform.name == "site_target"
    transform.fit(train_table())
    assert transform.overall == 4.6
    numpy.testing.assert_allclose(transform.table.loc[["s0", "s1"]].to_numpy(),
                                  [(9.0 + 2.0 * 4.6) / 5.0, (14.0 + 2.0 * 4.6) / 4.0], rtol=1e-12)
    out = transform.apply(held_table())
    assert list(out.columns) == ["site", "tier", "v", "y", "site_target"]
    numpy.testing.assert_allclose(out["site_target"].to_numpy(), [(14.0 + 9.2) / 4.0, 4.6, (9.0 + 9.2) / 5.0],
                                  rtol=1e-12)
    plain = build("/frame/kalfa/target_encoding", column="site", target="y", name="enc")
    assert plain.smoothing == 1.0
    plain.fit(train_table())
    numpy.testing.assert_allclose(plain.apply(held_table())["enc"].to_numpy(), [18.6 / 3.0, 4.6, 13.6 / 4.0],
                                  rtol=1e-12)
    zero = build("/frame/kalfa/target_encoding", column="site", target="y", smoothing=0)
    zero.fit(train_table())
    assert zero.table.to_dict() == {"s0": 3.0, "s1": 7.0}


def test_target_encoding_needs_a_table(root):
    transform = build("/frame/kalfa/target_encoding", column="site", target="y")
    with pytest.raises(ValueError, match=re.escape("target_encoding fits on a table in memory; a stream or a "
                                                   "Dataset source has no frame to fit on")):
        transform.fit(build("/source/kalfa/parquet_stream", path=str(root / "housing.parquet")))


def test_fit_frames_fits_each_transform_on_what_the_earlier_ones_produced_and_records_them(tmp_path):
    frames = [build("/frame/kalfa/group_statistic", by="site", column="v", statistic="median"),
              build("/frame/kalfa/target_encoding", column="v_median_by_site", target="y", smoothing=0.0)]
    fitted = build("/lego/kalfa/fit_frames", df=train_table(), frames=frames, record=str(tmp_path))
    assert fitted == frames
    assert fitted[0] is frames[0] and fitted[1] is frames[1]
    assert fitted[0].table.to_dict() == {"s0": 2.0, "s1": 7.0}
    assert fitted[1].table.to_dict() == {2.0: 3.0, 7.0: 7.0}
    assert fitted[1].overall == 4.6
    assert (tmp_path / "fitted" / "frames" / "frames.pkl").is_file()
    read = build("/lego/kalfa/read_frames", record=str(tmp_path))
    assert [type(item) for item in read] == [GroupStatistic, TargetEncoding]
    assert read[0] is not fitted[0]
    assert read[0].name == "v_median_by_site"
    assert read[0].table.to_dict() == {"s0": 2.0, "s1": 7.0}
    assert read[0].overall == 4.0
    assert read[1].name == "v_median_by_site_target"
    assert read[1].table.to_dict() == {2.0: 3.0, 7.0: 7.0}
    assert read[1].overall == 4.6


def test_apply_frames_applies_the_fitted_transforms_in_order_to_a_set(tmp_path):
    frames = [build("/frame/kalfa/group_statistic", by="site", column="v", statistic="median"),
              build("/frame/kalfa/target_encoding", column="v_median_by_site", target="y", smoothing=0.0)]
    fitted = build("/lego/kalfa/fit_frames", df=train_table(), frames=frames, record=str(tmp_path))
    out = build("/lego/kalfa/apply_frames", df=held_table(), frames=fitted)
    assert list(out.columns) == ["site", "tier", "v", "y", "v_median_by_site", "v_median_by_site_target"]
    assert out["v_median_by_site"].tolist() == [7.0, 4.0, 2.0]
    assert out["v_median_by_site_target"].tolist() == [7.0, 4.6, 3.0]
    assert out.index.tolist() == [7, 8, 9]
    read = build("/lego/kalfa/read_frames", record=str(tmp_path))
    again = build("/lego/kalfa/apply_frames", df=held_table(), frames=read)
    pandas.testing.assert_frame_equal(again, out)
    train = build("/lego/kalfa/apply_frames", df=train_table(), frames=fitted)
    assert train["v_median_by_site_target"].tolist() == [3.0, 3.0, 7.0, 3.0, 7.0]


def test_frames_without_transforms_pass_the_frame_untouched(tmp_path):
    df = train_table()
    assert build("/lego/kalfa/fit_frames", df=df, frames=[]) == []
    assert build("/lego/kalfa/fit_frames", df=df, frames=None) == []
    assert not (tmp_path / "fitted").exists()
    assert build("/lego/kalfa/apply_frames", df=df, frames=[]) is df
    assert build("/lego/kalfa/apply_frames", df=df, frames=None) is df
    assert build("/lego/kalfa/read_frames", record=str(tmp_path)) == []
    assert build("/lego/kalfa/fit_frames", df=df, frames=[], record=str(tmp_path)) == []
    assert (tmp_path / "fitted" / "frames" / "frames.pkl").is_file()
    assert build("/lego/kalfa/read_frames", record=str(tmp_path)) == []
