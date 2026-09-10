from pathlib import Path

import pandas
import pytest

from kalfa.api import check, predict
from kalfa.config import parse_sets
from kalfa.std.common.history import History
from kalfa.synthetic import anomaly_frame

pytestmark = pytest.mark.slow


def test_check_the_plugin_config(dataset):
    dataset("alad")
    prepared = check(["config.yaml"], parse_sets([]))
    assert [problem.kind for problem in prepared.problems] == ["unused_output"] * 4
    assert all(problem.severity == "warning" for problem in prepared.problems)
    assert prepared.sizes == {"train": 1400, "valid": 200, "test": 400}
    from kalfa.std.lego.kalfa.ratio_sizes import ratio_sizes

    kept = int((anomaly_frame()["is_anomaly"] != 5).sum())
    expected = ratio_sizes(kept, [0.7, 0.1, 0.2])
    loaded = check(["config.yaml"], parse_sets([]), load=True).loaded
    assert loaded["valid"] == expected["valid"] and loaded["test"] == expected["test"]
    assert 0 < loaded["train"] < expected["train"]
    document = prepared.document
    params = document["flow"]["data"]["params"]
    assert [step["params"]["query"] for step in params["transform_pre"]] == ["(is_anomaly > -4) & (is_anomaly < 4)"]
    assert [step["params"]["query"] for step in params["set_transforms"]["train"]["transforms"]] == ["is_anomaly == 0"]
    assert document["checkpoint"] == {}
    adv_d = document["losses"]["adv_d"]["params"]["objective"]
    assert adv_d["params"]["criterion"] == {"uri": "/criterion/kalfa/bce_logits"}


def test_two_optimizers_five_models_and_the_score_plots(trained):
    result = trained("alad")
    record = Path(result.record)
    assert record == Path("runs/alad_fixed")
    history = History.read(record)
    assert [line["turn"] for line in history] == [1, 2]
    assert {"train/adv_d", "train/adv_g", "val/adv_d", "val/adv_g", "val/auroc", "test/auroc", "val/average_precision",
            "test/average_precision", "lr/g", "lr/d"} <= set(history[0])
    assert 0.0 <= history[-1]["test/auroc"] <= 1.0
    assert not (record / "checkpoints").exists() and (record / "final" / "state.pt").exists()
    for name in ("plots/score_histogram.png", "plots/roc.png", "plots/pr_curve.png", "plots/modules.txt",
                 "plots/architecture_encoder.png", "plots/architecture_anomaly_score.png",
                 "predictions.parquet", "preprocessors/std_scaler.pkl", "preprocessors/log_scaler.pkl",
                 "preprocessors/plan.json", "resolved.yaml"):
        assert (record / name).exists(), name
    table = pandas.read_parquet(record / "predictions.parquet")
    assert list(table.columns) == ["row", "is_anomaly", "raw_score", "pred_score"]
    assert set(table["is_anomaly"].unique()) <= {0, 1} and (table["pred_score"] >= 0.0).all()
    again = predict(result.record)
    assert again.model == "anomaly_score" and len(again.table) == len(table)
