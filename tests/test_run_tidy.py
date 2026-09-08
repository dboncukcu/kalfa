"""The alad example end to end: the ALAD plugin, filters before and after the split, no checkpoint, four plots."""

from pathlib import Path

import pandas
import pytest

import kalfa  # noqa: F401
from conftest import ROOT
from kalfa.api import check, predict, run
from kalfa.config import parse_sets
from kalfa.record import read_history
from kalfa.synthetic import anomaly_frame, write_anomaly

CONFIG = str(ROOT / "examples" / "alad" / "config.yaml")


@pytest.fixture
def anomaly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_anomaly(tmp_path / "anomaly_data.parquet")
    return tmp_path


def test_check_tidy(anomaly):
    prepared = check([CONFIG], parse_sets([]))
    assert [problem.kind for problem in prepared.problems] == ["unused_output"] * 4
    assert all(problem.severity == "warning" for problem in prepared.problems)
    assert prepared.sizes == {"train": 1400, "valid": 200, "test": 400}
    from kalfa.std.split import sizes as split_sizes

    kept = int((anomaly_frame()["is_anomaly"] != 5).sum())
    expected = split_sizes(kept, [0.7, 0.1, 0.2])
    loaded = check([CONFIG], parse_sets([]), load=True).loaded
    assert loaded["valid"] == expected["valid"] and loaded["test"] == expected["test"]
    assert 0 < loaded["train"] < expected["train"]
    document = prepared.document
    assert document["flow"]["data"]["params"]["filter_pre"] == ["(is_anomaly > -4) & (is_anomaly < 4)"]
    assert document["flow"]["data"]["params"]["filter_set"] == [{"query": "is_anomaly == 0", "sets": ["train"]}]
    assert document["flow"]["training"]["params"]["checkpoint"] is None
    assert document["losses"]["adv_d"]["params"]["criterion"] == {"uri": "/criterion/kalfa/bce_logits"}


@pytest.mark.filterwarnings("ignore::cirak.errors.CirakWarning")      # the dxx heads write a feature the composite alone reads
@pytest.mark.filterwarnings("ignore:auroc is undefined")             # one class in a set of the tiny synthetic table
@pytest.mark.filterwarnings("ignore:average_precision is undefined")
def test_run_tidy(anomaly):
    result = run([CONFIG], parse_sets(params=["epochs=2"]), when="fixed")
    record = Path(result.record)
    assert record == Path("runs/alad_fixed")
    history = read_history(record)
    assert [line["turn"] for line in history] == [1, 2]
    assert {"train/adv_d", "train/adv_g", "val/adv_d", "val/adv_g", "val/auroc", "test/auroc", "val/average_precision",
            "test/average_precision", "lr/g", "lr/d"} <= set(history[0])
    assert 0.0 <= history[-1]["test/auroc"] <= 1.0
    assert not (record / "checkpoints").exists() and (record / "final" / "state.pt").exists()
    for name in ("plots/score_histogram.png", "plots/roc.png", "plots/pr_curve.png", "plots/architecture.txt",
                 "predictions.parquet", "preprocessors/std_scaler.pkl", "preprocessors/log_scaler.pkl",
                 "preprocessors/plan.json", "resolved.yaml"):
        assert (record / name).exists(), name
    table = pandas.read_parquet(record / "predictions.parquet")
    assert list(table.columns) == ["row", "is_anomaly", "raw_score", "pred_score"]
    assert set(table["is_anomaly"].unique()) <= {0, 1} and (table["pred_score"] >= 0.0).all()
    again = predict(result.record)
    assert again.model == "anomaly_score" and len(again.table) == len(table)
