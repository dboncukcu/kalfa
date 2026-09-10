"""The frame kind: fitted on the train set, applied to every set, kept in the record."""

import pandas
import pytest

import kalfa  # noqa: F401
from kalfa.std.frame.base import read_frames
from kalfa.std.frame.kalfa.group_statistic import GroupStatistic
from kalfa.std.frame.kalfa.target_encoding import TargetEncoding
from kalfa.std.lego.kalfa.apply_frames import apply_frames
from kalfa.std.lego.kalfa.fit_frames import fit_frames


def test_group_statistic_learns_on_train_and_maps_every_set():
    train = pandas.DataFrame({"g": ["a", "a", "b"], "v": [1.0, 3.0, 10.0]})
    statistic = GroupStatistic(by="g", column="v", statistic="mean")
    statistic.fit(train)
    assert statistic.name == "v_mean_by_g" and statistic.overall == pytest.approx(14.0 / 3)
    out = statistic.apply(pandas.DataFrame({"g": ["b", "a", "c"], "v": [0.0, 0.0, 0.0]}))
    assert out["v_mean_by_g"].tolist() == pytest.approx([10.0, 2.0, 14.0 / 3])
    two = GroupStatistic(by=["g", "h"], column="v", statistic="max", name="top")
    two.fit(train.assign(h=[1, 2, 1]))
    assert two.apply(train.assign(h=[1, 1, 1]))["top"].tolist() == [1.0, 1.0, 10.0]
    with pytest.raises(ValueError):
        GroupStatistic(by="g", column="v", statistic="mode")


def test_target_encoding_smooths_toward_the_overall_mean():
    train = pandas.DataFrame({"c": ["x", "x", "y"], "t": [1.0, 3.0, 8.0]})
    encoding = TargetEncoding(column="c", target="t", smoothing=0.0)
    encoding.fit(train)
    out = encoding.apply(pandas.DataFrame({"c": ["y", "x", "z"]}))
    assert out["c_target"].tolist() == pytest.approx([8.0, 2.0, 4.0])
    smooth = TargetEncoding(column="c", target="t", smoothing=1.0, name="enc")
    smooth.fit(train)
    assert smooth.apply(train)["enc"].tolist() == pytest.approx([(4.0 + 4.0) / 3, (4.0 + 4.0) / 3, (8.0 + 4.0) / 2])


def test_fit_frames_chains_them_and_the_record_reads_them_back(tmp_path):
    train = pandas.DataFrame({"g": ["a", "b", "b"], "v": [1.0, 2.0, 4.0], "t": [0.0, 1.0, 1.0]})
    frames = [GroupStatistic(by="g", column="v"), TargetEncoding(column="g", target="t", smoothing=0.0)]
    fitted = fit_frames(train, frames, record=str(tmp_path))
    assert [type(item).__name__ for item in fitted] == ["GroupStatistic", "TargetEncoding"]
    applied = apply_frames(pandas.DataFrame({"g": ["b"], "v": [0.0], "t": [0.0]}), fitted)
    assert applied["v_mean_by_g"].tolist() == [3.0] and applied["g_target"].tolist() == [1.0]
    again = read_frames(tmp_path)
    assert [item.name for item in again] == ["v_mean_by_g", "g_target"]
    assert read_frames(tmp_path / "nowhere") == [] and fit_frames(train, [], record=None) == []
    assert apply_frames(train, []) is train
