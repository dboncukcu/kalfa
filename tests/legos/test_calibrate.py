import json
import pickle

import numpy
import pandas
import pytest
import torch
from cirak.registry import registry

from helpers import build, frame, tiny_model
from kalfa.std import STD_URIS
from kalfa.std.calibrate.kalfa.threshold import Threshold
from kalfa.std.common.device import Device


CALIBRATIONS = sorted(uri for uri in STD_URIS if uri.startswith("/calibrate/"))


def summed_model(out_features=1):
    model = tiny_model(3, out_features)
    with torch.no_grad():
        model.nodes["layer"].weight.fill_(1.0)
        model.nodes["layer"].bias.fill_(0.0)
    return model


def loader_of(set_name="valid", rows=20, seed=3):
    dataset = build("/feed/kalfa/table", frame=frame(rows=rows, seed=seed, set_name=set_name))
    return build("/loader/kalfa/torch", data=dataset, set=set_name, eval_size=8)


def sums_of(loader):
    return loader.dataset.x.sum(dim=1).numpy()


def calibrate(models, loaders, calibrations, predicts="m", record=None, composites=None):
    return build("/lego/kalfa/calibrate", models=models, composites=composites or {}, prep=None, loaders=loaders,
                 calibrations=calibrations, predicts=predicts, record=record, device=None)


def test_calibrate_scope_is_the_threshold():
    assert CALIBRATIONS == ["/calibrate/kalfa/threshold"]
    assert registry.aliases()["threshold"] == "/calibrate/kalfa/threshold"
    assert registry.facts("/lego/kalfa/calibrate").returns == "calibrations"
    assert registry.facts("/lego/kalfa/calibrate").bus == {"record": "record", "device": "device"}


def test_threshold_defaults_to_the_valid_set_at_the_95th_quantile():
    threshold = build("/calibrate/kalfa/threshold")
    assert isinstance(threshold, Threshold)
    assert threshold.set == "valid" and threshold.quantile == 0.95 and threshold.output is None
    assert threshold.threshold is None and threshold.wire is None


def test_threshold_refuses_a_quantile_outside_the_unit_interval():
    with pytest.raises(ValueError, match=r"threshold.quantile must be between 0 and 1, got 1.5"):
        build("/calibrate/kalfa/threshold", quantile=1.5)
    with pytest.raises(ValueError, match=r"threshold.quantile must be between 0 and 1, got -0.1"):
        build("/calibrate/kalfa/threshold", quantile=-0.1)


def test_threshold_fit_reads_the_quantile_of_the_raw_output_on_its_set():
    threshold = build("/calibrate/kalfa/threshold", set="valid", quantile=0.9)
    loader = loader_of()
    threshold.fit({"m": summed_model()}, {"valid": loader, "test": loader_of("test", seed=4)}, None, Device.cpu(),
                  "m")
    assert threshold.wire == "y"
    assert threshold.threshold == pytest.approx(float(numpy.quantile(sums_of(loader), 0.9)))
    assert threshold.note() == {"set": "valid", "quantile": 0.9, "output": "y",
                                "threshold": pytest.approx(float(numpy.quantile(sums_of(loader), 0.9)))}


def test_threshold_fit_takes_the_first_column_of_the_named_output():
    threshold = build("/calibrate/kalfa/threshold", set="test", quantile=0.5, output="y")
    loader = loader_of("test")
    threshold.fit({"m": summed_model(out_features=2)}, {"test": loader}, None, Device.cpu(), "m")
    assert threshold.wire == "y"
    assert threshold.threshold == pytest.approx(float(numpy.quantile(sums_of(loader), 0.5)))


def test_threshold_fit_needs_its_set_and_a_known_output():
    threshold = build("/calibrate/kalfa/threshold", set="valid")
    with pytest.raises(ValueError, match=r"threshold reads the 'valid' set, which the run does not have"):
        threshold.fit({"m": summed_model()}, {"test": loader_of("test")}, None, Device.cpu(), "m")
    empty = build("/loader/kalfa/torch", data=build("/feed/kalfa/table", frame=frame(rows=0, set_name="valid")),
                  set="valid", eval_size=8)
    with pytest.raises(ValueError, match=r"threshold reads the 'valid' set, which the run does not have"):
        threshold.fit({"m": summed_model()}, {"valid": empty}, None, Device.cpu(), "m")
    named = build("/calibrate/kalfa/threshold", set="valid", output="z")
    with pytest.raises(KeyError, match=r"threshold names output 'z'; the model has \['y'\]"):
        named.fit({"m": summed_model()}, {"valid": loader_of()}, None, Device.cpu(), "m")


def test_threshold_apply_flags_the_rows_above_it():
    threshold = build("/calibrate/kalfa/threshold", quantile=0.5)
    table = pandas.DataFrame({"row": [0, 1, 2], "raw_y": [0.1, 0.5, 0.9], "pred_y": [1.0, 2.0, 3.0]})
    assert threshold.apply(table) is table
    threshold.fit({"m": summed_model()}, {"valid": loader_of()}, None, Device.cpu(), "m")
    threshold.threshold = 0.5
    flagged = threshold.apply(table)
    assert list(flagged.columns) == ["row", "raw_y", "pred_y", "flag_y"]
    assert flagged["flag_y"].tolist() == [False, False, True]
    assert "flag_y" not in table.columns
    other = pandas.DataFrame({"row": [0], "raw_z": [2.0]})
    assert threshold.apply(other) is other


def test_calibrate_step_fits_every_calibration_and_writes_the_record_files(tmp_path):
    loader = loader_of()
    hot = build("/calibrate/kalfa/threshold", set="valid", quantile=0.9)
    cold = build("/calibrate/kalfa/threshold", set="valid", quantile=0.1)
    fitted = calibrate({"m": summed_model()}, {"valid": loader}, {"hot": hot, "cold": cold}, record=str(tmp_path))
    assert fitted == {"hot": hot, "cold": cold}
    assert hot.threshold == pytest.approx(float(numpy.quantile(sums_of(loader), 0.9)))
    assert cold.threshold == pytest.approx(float(numpy.quantile(sums_of(loader), 0.1)))
    target = tmp_path / "fitted" / "calibrate"
    assert sorted(path.name for path in target.iterdir()) == ["calibrate.json", "calibrations.pkl"]
    note = json.loads((target / "calibrate.json").read_text())
    assert note == {"hot": hot.note(), "cold": cold.note()}
    with (target / "calibrations.pkl").open("rb") as stream:
        restored = pickle.load(stream)
    assert list(restored) == ["hot", "cold"]
    assert isinstance(restored["hot"], Threshold)
    assert restored["hot"].threshold == hot.threshold and restored["hot"].wire == "y"
    flagged = restored["hot"].apply(pandas.DataFrame({"raw_y": [hot.threshold - 1.0, hot.threshold + 1.0]}))
    assert flagged["flag_y"].tolist() == [False, True]


def test_calibrate_step_reaches_a_composite_predicts_model_and_writes_nothing_without_a_record(tmp_path):
    threshold = build("/calibrate/kalfa/threshold", set="valid", quantile=0.5)
    loader = loader_of()
    fitted = calibrate({}, {"valid": loader}, {"cut": threshold}, predicts="full", composites={"full": summed_model()})
    assert fitted == {"cut": threshold}
    assert threshold.threshold == pytest.approx(float(numpy.quantile(sums_of(loader), 0.5)))
    assert list(tmp_path.iterdir()) == []
    assert calibrate({"m": summed_model()}, {"valid": loader}, {}, record=str(tmp_path)) == {}
    assert json.loads((tmp_path / "fitted" / "calibrate" / "calibrate.json").read_text()) == {}
