import json
from pathlib import Path

import numpy
import pandas
import pytest

from data import reference_frame
from helpers import reference_sets


pytestmark = pytest.mark.slow

COLUMNS = ["row", "y_lin", "y_quad", "y_heavy", "y_frac", "is_hot", "raw_y_hat_0", "raw_y_hat_1", "pred_y_hat_y_lin",
           "pred_y_hat_y_quad", "raw_aux_hat_0", "raw_aux_hat_1", "pred_aux_hat_y_heavy", "pred_aux_hat_y_frac",
           "raw_tail_logit", "pred_tail_logit_is_hot", "sample_id", "site", "flag_tail_logit"]


def table_of(record):
    return pandas.read_parquet(Path(record) / "predictions.parquet")


def test_the_prediction_table_has_the_documented_columns_in_order(reference):
    table = table_of(reference.record)
    data = json.loads((Path(reference.record) / "data.json").read_text())
    assert list(table.columns) == COLUMNS
    assert len(table) < data["sets"]["test"]["rows"] == data["split"]["test"]


def test_the_rows_are_the_masked_test_rows_of_the_source(reference):
    table = table_of(reference.record)
    source = reference_frame()
    rows = table["row"].to_numpy()
    test = reference_sets()["test"]
    assert rows.tolist() == test.index[test["num_0"] <= 1.5].tolist()
    assert (source.loc[rows, "raw_0"] <= 1.5).all()
    assert (source.loc[rows, "raw_1"] > -2.5).all()
    assert (table["sample_id"].to_numpy() == rows + 1000).all()
    assert (table["site"].to_numpy() == source.loc[rows, "site"].to_numpy()).all()


def test_the_targets_are_inverted_back_to_the_source_scale(reference):
    table = table_of(reference.record)
    source = reference_frame()
    rows = table["row"].to_numpy()
    for name in ("y_lin", "y_quad", "y_frac"):
        numpy.testing.assert_allclose(table[name].to_numpy(), source.loc[rows, name].to_numpy(), rtol=1e-5,
                                      atol=1e-5)
    numpy.testing.assert_allclose(numpy.tanh(table["y_heavy"].to_numpy() / 3.0),
                                  numpy.tanh(source.loc[rows, "y_heavy"].to_numpy() / 3.0), atol=1e-6)
    assert (table["is_hot"].to_numpy() == source.loc[rows, "is_hot"].to_numpy()).all()


def test_the_predictions_are_decoded_through_the_target_chains(reference):
    table = table_of(reference.record)
    assert ((table["pred_aux_hat_y_frac"] > 0.0) & (table["pred_aux_hat_y_frac"] < 1.0)).all()
    assert numpy.isfinite(table["pred_aux_hat_y_heavy"].to_numpy()).all()
    numpy.testing.assert_allclose(numpy.tanh(table["pred_aux_hat_y_heavy"].to_numpy() / 3.0),
                                  table["raw_aux_hat_0"].to_numpy().clip(-1.0, 1.0), atol=1e-5)
    numpy.testing.assert_array_equal(table["pred_tail_logit_is_hot"].to_numpy(), table["raw_tail_logit"].to_numpy())
    assert table["pred_y_hat_y_lin"].std() > 0.0 and table["raw_y_hat_0"].std() > 0.0


def test_the_calibrated_flag_applies_the_fitted_threshold(reference):
    table = table_of(reference.record)
    threshold = json.loads((Path(reference.record) / "fitted/calibrate/calibrate.json").read_text())
    cut = threshold["hot_cut"]["threshold"]
    assert table["flag_tail_logit"].dtype == bool
    numpy.testing.assert_array_equal(table["flag_tail_logit"].to_numpy(), (table["raw_tail_logit"] > cut).to_numpy())
    assert 0 < table["flag_tail_logit"].sum() < len(table)
