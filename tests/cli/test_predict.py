import shutil
from pathlib import Path

import numpy
import pandas
import pytest

from kalfa.cli import main


COLUMNS = ["row", "y_lin", "y_quad", "y_heavy", "y_frac", "is_hot", "raw_y_hat_0", "raw_y_hat_1", "pred_y_hat_y_lin",
           "pred_y_hat_y_quad", "raw_aux_hat_0", "raw_aux_hat_1", "pred_aux_hat_y_heavy", "pred_aux_hat_y_frac",
           "raw_tail_logit", "pred_tail_logit_is_hot", "sample_id", "site", "flag_tail_logit"]
TOWER = ["row", "y_lin", "y_quad", "y_heavy", "y_frac", "is_hot", *[f"raw_h_{index}" for index in range(16)],
         "sample_id", "site"]


@pytest.fixture
def copy(reference, tmp_path):
    target = tmp_path / "ref"
    shutil.copytree(reference.record, target)
    return str(target)


def test_predict_rewrites_the_predictions_of_the_test_set(copy, reference, capsys):
    before = pandas.read_parquet(Path(reference.record) / "predictions.parquet")
    (Path(copy) / "predictions.parquet").unlink()
    assert main(["predict", copy]) == 0
    assert capsys.readouterr().out == f"predicted 278 rows with full: {copy}/predictions.parquet\n"
    after = pandas.read_parquet(Path(copy) / "predictions.parquet")
    assert list(after.columns) == COLUMNS and len(after) == 278
    pandas.testing.assert_frame_equal(after, before)


def test_predict_data_writes_a_file_named_after_the_data(copy, capsys):
    assert main(["predict", copy, "--data", "new.parquet"]) == 0
    assert capsys.readouterr().out == f"predicted 95 rows with full: {copy}/predictions_new.parquet\n"
    table = pandas.read_parquet(Path(copy) / "predictions_new.parquet")
    assert list(table.columns) == COLUMNS and len(table) == 95
    assert table["sample_id"].between(1000, 1099).all() and table["flag_tail_logit"].dtype == bool
    assert len(pandas.read_parquet(Path(copy) / "predictions.parquet")) == 278


def test_predict_model_reaches_any_model_and_its_ema_copy(copy, capsys):
    assert main(["predict", copy, "--model", "tower"]) == 0
    assert capsys.readouterr().out == f"predicted 278 rows with tower: {copy}/predictions_tower.parquet\n"
    assert main(["predict", copy, "--model", "tower.ema"]) == 0
    assert capsys.readouterr().out == f"predicted 278 rows with tower.ema: {copy}/predictions_tower.ema.parquet\n"
    raw = pandas.read_parquet(Path(copy) / "predictions_tower.parquet")
    ema = pandas.read_parquet(Path(copy) / "predictions_tower.ema.parquet")
    assert list(raw.columns) == list(ema.columns) == TOWER and len(raw) == len(ema) == 278
    assert not numpy.allclose(raw["raw_h_0"], ema["raw_h_0"])
    assert main(["predict", copy, "--model", "ghost"]) == 1
    assert "unknown model 'ghost'; the models are ['full', 'head_aux', 'head_lin', 'lambdas', 'tail_head', " \
           "'tail_stem', 'tower']" in capsys.readouterr().err
    assert main(["predict", copy, "--model", "head_lin.ema"]) == 1
    assert "model 'head_lin' has no ema copy" in capsys.readouterr().err


@pytest.mark.parametrize("which", ["best", "last", "final"])
def test_predict_which_loads_the_named_weights(copy, which, capsys):
    assert main(["predict", copy, "--which", which, "--log"]) == 0
    captured = capsys.readouterr()
    assert captured.out == f"predicted 278 rows with full: {copy}/predictions.parquet\n"
    assert f"  INFO   run             predicting with full ({which} weights)\n" in captured.err
    assert f"  INFO   run             278 rows -> {copy}/predictions.parquet\n" in captured.err


def test_predict_plots_draws_the_named_plots_with_the_suffix_of_the_data(copy, capsys):
    assert main(["predict", copy, "--data", "new.parquet", "--plots", "pred_vs_true,binary_roc"]) == 0
    assert capsys.readouterr().out == (f"predicted 95 rows with full: {copy}/predictions_new.parquet\n"
                                       f"plots pred_vs_true, binary_roc: {copy}/plots\n")
    drawn = sorted(path.name for path in (Path(copy) / "plots").iterdir() if path.name.endswith("_new.png"))
    assert drawn == ["binary_roc_new.png", "pred_vs_true_new.png"]
    assert main(["predict", copy, "--plots", "nope"]) == 1
    assert capsys.readouterr().err.startswith("plots ['nope'] are not in the plots section, which has ['architecture', "
                                              "'architecture_text', 'binary_precision_recall_curve', 'binary_roc', ")


def test_predict_device_names_a_device_lego(copy, capsys):
    assert main(["predict", copy, "--device", "cpu"]) == 0
    assert capsys.readouterr().out == f"predicted 278 rows with full: {copy}/predictions.parquet\n"
    assert main(["predict", copy, "--device", "nope"]) == 1
    assert capsys.readouterr().err == ("device 'nope' is not a known device lego; the short names are "
                                       "['auto', 'cpu', 'cuda', 'mps']\n")


def test_generate_needs_a_generate_section(copy, capsys):
    assert main(["generate", copy]) == 1
    assert capsys.readouterr().err == f"{copy}: the config has no generate section\n"
    assert not (Path(copy) / "samples").exists()
