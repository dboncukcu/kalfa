"""Config 02 end to end on a synthetic churn table: labels decoded, class weights built at run time, best by val/f1."""

import math
from pathlib import Path

import pandas
import pytest
import torch

import kalfa  # noqa: F401
from conftest import ROOT
from kalfa.api import check, predict, run
from kalfa.config import parse_sets
from kalfa.record import read_history
from kalfa.synthetic import write_churn

CONFIG = str(ROOT / "configs" / "02_mlp_classification.yaml")


@pytest.fixture
def churn(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_churn(tmp_path / "churn.parquet")
    return tmp_path


def test_check_reads_the_header_and_accepts_the_chains(churn):
    prepared = check([CONFIG])
    assert prepared.problems == []
    assert prepared.sizes == {"train": 1400, "valid": 300, "test": 300}


def test_run_02(churn):
    result = run([CONFIG], parse_sets(params=["epochs=3"]), when="fixed")
    record = Path(result.record)
    history = read_history(record)
    assert [line["turn"] for line in history] == [1, 2, 3]
    keys = set(history[0])
    assert {"train/ce", "train/accuracy", "train/f1", "val/ce", "val/accuracy", "val/f1", "test/f1", "lr/net"} <= keys
    assert all(math.isfinite(line["train/ce"]) for line in history)
    assert 0.5 < history[-1]["val/accuracy"] <= 1.0
    payload = torch.load(record / "checkpoints" / "best.pt", weights_only=False)
    assert payload["checkpoint"]["best"] == max(line["val/f1"] for line in history)
    predictions = pandas.read_parquet(record / "predictions.parquet")
    assert list(predictions.columns) == ["row", "churned", "raw_logits_0", "raw_logits_1", "pred_logits"]
    assert set(predictions["churned"]) <= {"yes", "no"} and set(predictions["pred_logits"]) <= {"yes", "no"}
    source = pandas.read_parquet(churn / "churn.parquet")
    assert predictions["churned"].tolist() == source["churned"].iloc[predictions["row"]].tolist()
    agreement = float((predictions["churned"] == predictions["pred_logits"]).mean())
    assert agreement > 0.6
    assert (record / "plots" / "confusion.png").exists() and (record / "plots" / "loss_curve.png").exists()
    plan = (record / "preprocessors" / "plan.json").read_text()
    assert "cat_a_north" in plan and "label" in plan
    write_churn(churn / "new.parquet", rows=25, seed=9)
    prediction = predict(result.record, data=str(churn / "new.parquet"))
    assert len(prediction.table) == 25 and set(prediction.table["pred_logits"]) <= {"yes", "no"}
    assert list(prediction.table.columns) == ["row", "churned", "raw_logits_0", "raw_logits_1", "pred_logits"]


def test_class_weights_and_encoders(churn):
    from kalfa.std.data import class_weights
    from kalfa.std.feed import table
    from kalfa.std.loader import torch as torch_loader
    from kalfa.std.pre import apply, fit, label_encoder, one_hot, standard_scaler

    data = pandas.read_parquet(churn / "churn.parquet")
    prep = fit(data, {"num_*": {"preprocessors": ["s"]}, "cat_*": {"preprocessors": ["o"]},
                      "churned": {"target": True, "preprocessors": ["l"]}},
               {"s": standard_scaler(), "o": one_hot(), "l": label_encoder()}, [])
    assert [column for column in prep.features if column.startswith("cat_a")] == ["cat_a_east", "cat_a_north",
                                                                                 "cat_a_south"]
    frame = apply(data, prep, "train")
    assert frame.data["churned"].dtype == "int64" and set(frame.data["churned"]) == {0, 1}
    loader = torch_loader(table(frame), "train", {"size": 64})
    weights = class_weights(loader)
    assert weights.shape == (2,) and abs(float(weights.mean()) - 1.0) < 1e-5
    assert prep.inverse("churned", [0, 1]).tolist() == ["no", "yes"]
    assert prep.decode("churned", [[2.0, 0.5], [0.1, 0.9]]).tolist() == ["no", "yes"]
    assert prep.decode("churned", [[-1.0], [1.0]]).tolist() == ["no", "yes"]
    with pytest.raises(ValueError, match="not seen"):
        prep.fitted["l"]["churned"].apply(["maybe"])
