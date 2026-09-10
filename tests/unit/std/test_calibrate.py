"""The mask on the table frame and the calibrate kind."""

import json

import numpy
import pandas
import pytest

import kalfa  # noqa: F401
from helpers import tiny_model
from kalfa.std.calibrate.base import read_calibrations
from kalfa.std.calibrate.kalfa.threshold import Threshold
from kalfa.std.common.device import Device
from kalfa.std.feed.kalfa.table import table
from kalfa.std.lego.kalfa.apply import apply
from kalfa.std.lego.kalfa.calibrate import calibrate
from kalfa.std.lego.kalfa.fit import fit
from kalfa.std.lego.kalfa.predict import predict
from kalfa.std.loader.kalfa.torch import torch_loader
from kalfa.std.plot.base import set_frame
from kalfa.synthetic import housing_frame


def fitted(rows=40):
    df = housing_frame(rows=rows, columns=3)
    prep = fit(df, {"x*": {}, "price": {"target": True}}, {}, [])
    return df, prep


def test_the_mask_keeps_rows_in_the_frame_and_out_of_the_loader():
    df, prep = fitted()
    frame = apply(df, prep, "test", mask="x0 > 0")
    kept = int((df["x0"] <= 0).sum())
    assert frame.mask.sum() == kept and len(frame.data) == 40
    dataset = table(frame)
    assert len(dataset) == kept and dataset.rows().tolist() == df.index[df["x0"] <= 0].tolist()
    loader = torch_loader(dataset, "test", 8)
    assert sum(len(batch["x"]) for batch in loader) == kept
    view = set_frame({"test": loader}, prep, "test")
    assert len(view) == 40 and view["masked"].sum() == 40 - kept
    plain = apply(df, prep, "test")
    assert plain.mask is None
    assert "masked" not in set_frame({"test": torch_loader(table(plain), "test", 8)}, prep, "test")


def test_threshold_reads_a_quantile_off_the_set_and_flags_the_predictions(tmp_path):
    df, prep = fitted()
    model = tiny_model()
    loaders = {name: torch_loader(table(apply(df, prep, name)), name, 8) for name in ("valid", "test")}
    threshold = Threshold(set="valid", quantile=0.5)
    threshold.fit({"m": model}, loaders, prep, Device.cpu(), "m")
    assert threshold.wire == "y" and threshold.note()["set"] == "valid"
    predictions = predict({"m": model}, {}, loaders["test"], prep, "m", "test", calibrations={"cut": threshold})
    assert "flag_y" in predictions and 0 < int(predictions["flag_y"].sum()) < 40
    assert predictions["flag_y"].tolist() == (predictions["raw_y"] > threshold.threshold).tolist()
    items = calibrate({"m": model}, {}, prep, loaders, {"cut": Threshold(set="test", quantile=0.9)}, "m",
                      record=str(tmp_path), device=Device.cpu())
    assert list(items) == ["cut"] and items["cut"].threshold is not None
    again = read_calibrations(tmp_path)
    assert again["cut"].threshold == items["cut"].threshold
    assert json.loads((tmp_path / "fitted" / "calibrate" / "calibrate.json").read_text())["cut"]["quantile"] == 0.9
    assert read_calibrations(tmp_path / "nowhere") == {} and calibrate({}, {}, prep, loaders, {}, "m") == {}
    with pytest.raises(ValueError):
        Threshold(set="valid", quantile=2.0)
    with pytest.raises(ValueError):
        Threshold(set="train").fit({"m": model}, loaders, prep, Device.cpu(), "m")
    assert isinstance(threshold.apply(pandas.DataFrame({"row": [0]})), pandas.DataFrame)
    assert numpy.asarray(threshold.apply(pandas.DataFrame({"raw_y": [1e9]}))["flag_y"]).tolist() == [True]
