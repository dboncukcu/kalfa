import math
from pathlib import Path

import pandas
import pytest
import torch

from kalfa.api import check, predict

pytestmark = pytest.mark.slow


def test_check_reads_the_header_and_accepts_the_chains(dataset):
    dataset("02_mlp_classification")
    prepared = check(["config.yaml"])
    assert prepared.problems == []
    assert prepared.sizes == {"train": 1400, "valid": 300, "test": 300}


def test_labels_are_decoded_and_the_best_is_by_f1(trained):
    from kalfa.std.common.history import History

    result = trained("02_mlp_classification")
    record = Path(result.record)
    history = History.read(record)
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
    source = pandas.read_parquet("churn.parquet")
    assert predictions["churned"].tolist() == source["churned"].iloc[predictions["row"]].tolist()
    agreement = float((predictions["churned"] == predictions["pred_logits"]).mean())
    assert agreement > 0.6
    assert (record / "plots" / "confusion.png").exists() and (record / "plots" / "loss_curve.png").exists()
    plan = (record / "preprocessors" / "plan.json").read_text()
    assert "cat_a_north" in plan and "label" in plan
    prediction = predict(result.record, data="new.parquet")
    assert len(prediction.table) == 100 and set(prediction.table["pred_logits"]) <= {"yes", "no"}
    assert list(prediction.table.columns) == ["row", "churned", "raw_logits_0", "raw_logits_1", "pred_logits"]


def test_class_weights_and_encoders(dataset):
    from kalfa.std.data.kalfa.class_weights import class_weights
    from kalfa.std.feed.kalfa.table import table
    from kalfa.std.loader.kalfa.torch import torch_loader
    from kalfa.std.lego.kalfa.apply import apply
    from kalfa.std.lego.kalfa.fit import fit
    from kalfa.std.pre.kalfa.label_encoder import LabelEncoder
    from kalfa.std.pre.kalfa.one_hot import OneHot
    from kalfa.std.pre.sklearn.standard_scaler import StandardScaler

    dataset("02_mlp_classification")
    data = pandas.read_parquet("churn.parquet")
    prep = fit(data, {"num_*": {"preprocessors": ["s"]}, "cat_*": {"preprocessors": ["o"]},
                      "churned": {"target": True, "preprocessors": ["l"]}},
               {"s": StandardScaler(), "o": OneHot(), "l": LabelEncoder()}, [])
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
