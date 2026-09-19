import json
from pathlib import Path

import pandas
import pytest

from helpers import config_path, copied
from kalfa.api import check, predict
from kalfa.config import parse_sets
from kalfa.std.common.history import History


pytestmark = pytest.mark.slow

REFUSALS = [("data.split={uri: random_split, params: {ratios: [0.7, 0.15, 0.15], seed: 1}}", "lazy_split"),
            ("data.batch={size: 64, balanced: true}", "lazy_batch"),
            ("data.feed={uri: window, params: {size: 4, horizon: 1}}", "lazy_feed"),
            ("data.frame=[{uri: group_statistic, params: {by: x0, column: x1}}]", "lazy_frame"),
            ("data.mask=x0 > 0", "lazy_mask"),
            ("data.spectators=[x0]", "lazy_spectators"),
            ("data.transform=[{uri: derive, params: {column: z, expr: x0 + x1}}]", "lazy_transform"),
            ("losses.loss={uri: cross_entropy, params: {weight: {uri: class_weights}}}", "lazy_data")]


@pytest.fixture
def stream(trained, at_root):
    return trained("stream")


def test_the_lazy_set_is_measured_by_streaming_and_refuses_what_needs_the_table(at_root):
    prepared = check([config_path("stream")], parse_sets([]), measure=True)
    assert prepared.problems == []
    assert prepared.sizes == prepared.measured == {"train": 1400, "valid": 300, "test": 300}
    for setting, kind in REFUSALS:
        found = check([config_path("stream")], parse_sets([setting]))
        assert kind in [problem.kind for problem in found.errors], setting
    warned = check([config_path("stream")], parse_sets(["data.preprocessors.enc={uri: label_encoder}",
                                                        "data.fields.x0={preprocessors: [enc]}"]))
    assert [problem.kind for problem in warned.warnings] == ["lazy_fit"]


def test_the_stream_trains_and_predicts_in_file_order(stream):
    history = History.read(stream.record)
    assert [list(line) for line in history] == [["turn", "global_step", "train/loss", "train/rmse", "val/loss",
                                                 "val/rmse", "test/loss", "test/rmse", "lr/model", "minimizes/model",
                                                 "seconds", "rules"]] * 2
    assert [line["global_step"] for line in history] == [22, 44]
    data = json.loads((Path(stream.record) / "data.json").read_text())
    assert data["split"] == {"test": None, "train": None, "valid": None}
    assert data["loaders"]["train"] == {"batches": None, "size": 64}
    table = pandas.read_parquet(Path(stream.record) / "predictions.parquet")
    assert list(table.columns) == ["row", "price", "raw_y", "pred_y"]
    assert table["row"].tolist() == list(range(1700, 2000))


def test_predict_streams_a_new_file(stream, tmp_path):
    copy = copied(stream.record, tmp_path / "stream_copy")
    result = predict(copy, data="new.csv")
    assert result.path == str(copy / "predictions_new.parquet")
    assert result.table["row"].tolist() == list(range(50))
