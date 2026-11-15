import json
import threading
import urllib.request

import numpy
import pandas
import pytest

from kalfa.board.answers import Board
from kalfa.board.server import make_server
from kalfa.core.errors import KalfaError
from kalfa.records import stats


def plan_with(pairs, targets):
    nodes = {"data": {"slot": "data", "use": "kalfa.std.data.table.Table",
                      "params": {"targets": {name: None for name in targets}}, "keys": {}}}
    for index, (pred, target) in enumerate(pairs):
        nodes[f"metrics.m{index}"] = {"slot": "metric", "use": "kalfa.std.metrics.rmse", "params": {},
                                      "keys": {"pred": pred, "target": target}}
    return {"format": 1, "nodes": nodes, "data": {"inputs": ["x"], "fields": {}, "classes": None}}


@pytest.fixture
def root(tmp_path):
    generator = numpy.random.default_rng(0)
    count = 120
    frame = pandas.DataFrame({"row": numpy.arange(count), "price": generator.normal(10, 2, count),
                              "flag": generator.integers(0, 2, count)})
    frame["pred_net"] = frame["price"] + generator.normal(0, 0.5, count)
    frame["pred_y_flag"] = numpy.clip(frame["flag"] * 0.6 + generator.random(count) * 0.4, 0, 1)
    for index in range(4):
        frame[f"pred_z_{index}"] = generator.normal(size=count)
    directory = tmp_path / "runs" / "rec"
    directory.mkdir(parents=True)
    frame.to_parquet(directory / "predictions.parquet", index=False)
    plan = plan_with([("net", "price"), ("y_flag", "flag")], ["price", "flag"])
    (directory / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    return tmp_path / "runs"


def pair_of(answer, pred):
    return next(pair for pair in answer["pairs"] if pair["pred"] == pred)


def width(found):
    return len(found["counts"])


def test_each_histogram_takes_its_own_bins(root):
    directory = root / "rec"
    answer = stats.prediction_stats(directory, overrides={"pred_net": {"bins": {"residual_hist": 7}}})
    found = pair_of(answer, "pred_net")["stats"]
    assert width(found["residual_hist"]) == 7 and width(found["truth_hist"]) == 40 and width(found["pred_hist"]) == 40
    shared = pair_of(stats.prediction_stats(directory, overrides={"pred_net": {"bins": {"truth_hist": 5}}}),
                     "pred_net")["stats"]
    assert width(shared["truth_hist"]) == width(shared["pred_hist"]) == 5
    assert shared["truth_hist"]["edges"] == shared["pred_hist"]["edges"] and width(shared["residual_hist"]) == 40
    binary = pair_of(stats.prediction_stats(directory, overrides={"pred_y_flag": {"bins": {"score_hist": 9}}}),
                     "pred_y_flag")["stats"]
    assert len(binary["score_hist"]["positive"]) == len(binary["score_hist"]["negative"]) == 9
    embedding = pair_of(stats.prediction_stats(directory, overrides={"pred_z": {"bins": {"norm_hist": 6}}}),
                        "pred_z")["stats"]
    assert width(embedding["norm_hist"]) == 6
    other = pair_of(stats.prediction_stats(directory, overrides={"pred_net": {"bins": {"residual_hist": 7}}}),
                    "pred_y_flag")["stats"]
    assert len(other["score_hist"]["positive"]) == 40


def test_wrong_bins_are_refused_or_clamped(root):
    directory = root / "rec"
    with pytest.raises(KalfaError):
        stats.prediction_stats(directory, overrides={"pred_net": {"bins": {"score_hist": 9}}})
    with pytest.raises(KalfaError):
        stats.prediction_stats(directory, overrides={"pred_net": {"bins": {"residual_hist": 0}}})
    clamped = pair_of(stats.prediction_stats(directory, overrides={"pred_net": {"bins": {"residual_hist": 5000}}}),
                      "pred_net")["stats"]
    assert width(clamped["residual_hist"]) == 1000


def test_the_board_and_the_server_take_bins_per_histogram(root):
    board = Board(root)
    found = pair_of(board.predictions("rec", pair="pred_net", histograms={"residual_hist": 7, "truth_hist": 5}),
                    "pred_net")["stats"]
    assert width(found["residual_hist"]) == 7 and width(found["truth_hist"]) == 5
    with pytest.raises(KalfaError):
        board.predictions("rec", histograms={"residual_hist": 7})
    httpd = make_server(root, host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        query = "pair=pred_net&histograms=residual_hist:7,truth_hist:5"
        with urllib.request.urlopen(f"{base}/api/record/rec/predictions?{query}", timeout=10) as response:
            served = pair_of(json.loads(response.read()), "pred_net")["stats"]
        assert width(served["residual_hist"]) == 7 and width(served["truth_hist"]) == 5
    finally:
        httpd.shutdown()
        httpd.server_close()
