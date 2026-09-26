import json
import shutil
from pathlib import Path

import numpy
import pandas
import pytest
import torch

from data import reference_frame
from helpers import config_path, legacy_onnx, needs
from kalfa.api import check, export, plots, predict, prepare_data, resume, run, stop
from kalfa.collect import collect
from kalfa.config import parse_sets
from kalfa.errors import KalfaError
from kalfa.record import resume_chain
from kalfa.std.common.history import History


pytestmark = pytest.mark.slow


@pytest.fixture
def copy(reference, tmp_path):
    target = tmp_path / "ref_copy"
    shutil.copytree(reference.record, target)
    return target


def without_seconds(record):
    return [{key: value for key, value in line.items() if key != "seconds"} for line in History.read(record)]


def test_predict_rewrites_the_same_table(copy):
    before = pandas.read_parquet(copy / "predictions.parquet")
    result = predict(copy)
    assert result.path == str(copy / "predictions.parquet") and result.model == "full" and result.plots == []
    pandas.testing.assert_frame_equal(result.table, before)
    pandas.testing.assert_frame_equal(pandas.read_parquet(result.path), before)


def test_predict_on_new_data_replays_the_pre_split_steps_and_the_mask(copy):
    new = reference_frame(rows=100, seed=5)
    result = predict(copy, data="new.parquet")
    assert result.path == str(copy / "predictions_new.parquet")
    assert list(result.table.columns) == list(pandas.read_parquet(copy / "predictions.parquet").columns)
    assert len(result.table) == int(((new["raw_1"] > -2.5) & (new["raw_0"] <= 1.5)).sum())
    assert (result.table["sample_id"].to_numpy() == result.table["row"].to_numpy() + 1000).all()


def test_predict_with_a_frame_and_another_model_writes_its_raw_outputs(copy):
    result = predict(copy, data=reference_frame(rows=40, seed=3), model="tower")
    assert result.path == str(copy / "predictions_frame_tower.parquet")
    raw = [name for name in result.table.columns if name.startswith("raw_")]
    assert raw == [f"raw_h_{position}" for position in range(16)]
    assert [name for name in result.table.columns if name.startswith("pred_")] == []


def test_predict_reaches_the_ema_copy_by_name(copy):
    live = predict(copy, model="tower").table
    ema = predict(copy, model="tower.ema").table
    assert ema.shape == live.shape and list(ema.columns) == list(live.columns)
    assert not torch.equal(torch.tensor(ema["raw_h_0"].to_numpy()), torch.tensor(live["raw_h_0"].to_numpy()))


def test_export_writes_the_documented_formats(copy):
    exported = export(copy)
    assert exported.path == str(copy / "export" / "full.pt") and exported.format == "/export/kalfa/state_dict"
    assert isinstance(torch.load(exported.path, weights_only=False), dict)
    program = export(copy, format="pt2", model="tower")
    assert program.path == str(copy / "export" / "tower.pt2")
    assert torch.export.load(program.path) is not None
    with pytest.raises(KalfaError, match="is no export lego"):
        export(copy, format="pickle")


def test_export_to_onnx_needs_the_library(copy):
    needs("onnx")
    with legacy_onnx():
        exported = export(copy, format="onnx", out=copy / "onnx")
    assert exported.path == str(copy / "onnx" / "full.onnx") and Path(exported.path).stat().st_size > 0


def test_plots_redraw_the_named_plots_only(copy):
    (copy / "plots" / "loss_curve.png").unlink()
    (copy / "plots" / "binary_roc.png").unlink()
    drawn = plots(copy, only=["loss_curve", "pred_vs_true"])
    assert drawn.names == ["loss_curve", "pred_vs_true"]
    assert (copy / "plots" / "loss_curve.png").exists() and not (copy / "plots" / "binary_roc.png").exists()
    with pytest.raises(KalfaError, match="plots \\['nope'\\] are not in the plots section"):
        plots(copy, only=["nope"])


def test_resume_continues_from_the_last_checkpoint(copy):
    result = resume(copy, parse_sets(["training.epochs=5"]), when="resumed")
    assert result.record == "runs/ref_resumed"
    history = History.read(result.record)
    before = History.read(copy)
    assert [line["turn"] for line in history] == [4, 5]
    per_turn = before[0]["global_step"]
    assert [line["global_step"] for line in history] == [4 * per_turn, 5 * per_turn]
    assert json.loads((Path(result.record) / "resume.json").read_text()) == {
        "resume_from": str(copy), "checkpoint": str(copy / "checkpoints" / "last.pt")}
    assert resume_chain(result.record) == [str(copy)]
    assert (Path(result.record) / "checkpoints" / "best.pt").exists()
    assert history[-1]["lr/aux"] < before[-1]["lr/aux"]
    assert history[0]["rules"] == ["cool_stem"]


def test_resume_refuses_a_directory_without_a_state(tmp_path):
    with pytest.raises(KalfaError, match="nothing to resume"):
        resume(tmp_path / "nowhere")


def test_stop_refuses_a_finished_record(reference):
    with pytest.raises(KalfaError, match="has ended already"):
        stop(reference.record)


def test_collect_tabulates_a_single_record(copy, tmp_path):
    kind, text, target, written = collect([str(copy)])
    assert kind == "sweep" and target == str(copy / "reports") and written == ["sweep.csv", "sweep.json", "sweep.md"]
    assert text.startswith("── RUNS") and "test/rmse_lin" in text
    rows = json.loads((copy / "reports" / "sweep.json").read_text())["rows"]
    assert len(rows) == 1 and rows[0]["turns"] == 3 and rows[0]["dir"] == str(copy)


def test_a_run_from_prepared_data_matches_the_reference(reference, tmp_path):
    prepared = prepare_data([config_path("reference")], parse_sets([]), out=tmp_path / "prepared")
    directory = Path(prepared.directory)
    assert directory == tmp_path / "prepared"
    assert {path.name for path in directory.iterdir()} >= {"manifest.json", "train.parquet", "valid.parquet",
                                                            "test.parquet", "data.json", "fitted"}
    assert prepared.manifest["kind"] == "data" and prepared.sizes == {name: prepared.manifest["sizes"][name]
                                                                       for name in ("train", "valid", "test")}
    assert check([config_path("reference")], parse_sets([]), prepared=str(directory)).problems == []
    mismatch = check([config_path("reference")], parse_sets(["data.batch.size=32"]), prepared=str(directory))
    assert [problem.kind for problem in mismatch.errors] == ["prepared_mismatch"]
    result = run([config_path("reference")], parse_sets([]), when="from_prepared", prepared=str(directory))
    assert result.record == "runs/ref_from_prepared"
    assert without_seconds(result.record) == without_seconds(reference.record)
    assert json.loads((Path(result.record) / "manifest.json").read_text())["prepared"] == str(directory)


def test_report_best_reaches_a_composite_predicts(trained, at_root, tmp_path):
    worst = trained("reference", when="worst", sets=["training.checkpoint.params.mode=max"])
    best = torch.load(Path(worst.record) / "checkpoints" / "best.pt", weights_only=False)
    history = History.read(worst.record)
    values = [line["val/rmse_lin"] for line in history]
    assert best["turn"] == values.index(max(values)) + 1 < 3
    target = tmp_path / "worst_copy"
    shutil.copytree(worst.record, target)
    recorded = pandas.read_parquet(target / "predictions.parquet")
    pandas.testing.assert_frame_equal(predict(target, which="best").table, recorded)
    last = predict(target, which="last").table
    assert not numpy.allclose(last["raw_y_hat_0"].to_numpy(), recorded["raw_y_hat_0"].to_numpy())
