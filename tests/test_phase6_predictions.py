import json
import threading
import urllib.request

import numpy
import pandas
import pytest
from PIL import Image

from kalfa.board.answers import Board, jsonable
from kalfa.board.server import make_server
from kalfa.core.errors import KalfaError
from kalfa.records import stats


def plan_with(pairs, targets, classes=None):
    nodes = {"data": {"slot": "data", "use": "kalfa.std.data.table.Table",
                      "params": {"targets": {name: None for name in targets}}, "keys": {}}}
    for index, (pred, target) in enumerate(pairs):
        nodes[f"metrics.m{index}"] = {"slot": "metric", "use": "kalfa.std.metrics.rmse", "params": {},
                                      "keys": {"pred": pred, "target": target}}
    return {"format": 1, "nodes": nodes, "data": {"inputs": ["x"], "fields": {}, "classes": classes}}


def write_table(directory, frame, plan=None, name="predictions.parquet"):
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(directory / name, index=False)
    if plan is not None:
        (directory / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    return directory


def strict(value):
    return json.dumps(jsonable(value), allow_nan=False)


def by_pred(answer):
    return {pair["pred"]: pair for pair in answer["pairs"]}


def anomaly_table(directory):
    generator = numpy.random.default_rng(0)
    truth = numpy.r_[numpy.zeros(80, dtype=int), numpy.ones(20, dtype=int)]
    score = numpy.where(truth == 1, generator.normal(3.0, 1.0, 100), generator.normal(0.0, 1.0, 100))
    frame = pandas.DataFrame({"row": numpy.arange(100), "is_anomaly": truth, "pred_score": score.astype("float32")})
    write_table(directory, frame, plan_with([("score", "is_anomaly")], ["is_anomaly"]))
    return truth, frame["pred_score"].to_numpy(dtype=numpy.float64)


def test_a_two_valued_numeric_target_with_a_score_is_binary(tmp_path):
    truth, values = anomaly_table(tmp_path / "rec")
    answer = stats.prediction_stats(tmp_path / "rec")
    strict(answer)
    [pair] = answer["pairs"]
    assert pair["auto"] == pair["task"] == "binary" and pair["tasks"] == ["binary", "regression"]
    assert pair["positive"] == "1" and pair["score"] == "pred_score" and pair["columns"] == ["pred_score"]
    found = pair["stats"]
    assert found["labels"] == ["0", "1"] and found["threshold_rule"] == "best_f1"
    (tn, fp), (fn, tp) = found["confusion"]
    assert tn + fp + fn + tp == 100 and tp + fn == 20
    predicted = values >= found["threshold"]
    assert tp == int((predicted & (truth == 1)).sum()) and fp == int((predicted & (truth == 0)).sum())
    assert found["recall"] == pytest.approx(tp / 20) and found["precision"] == pytest.approx(tp / (tp + fp))
    assert found["f1"] == pytest.approx(2 * tp / (2 * tp + fp + fn))
    assert found["specificity"] == pytest.approx(tn / 80)
    assert found["roc"]["auc"] > 0.9 and found["pr"]["ap"] > 0.7
    assert 2 <= len(found["sweep"]["threshold"]) <= 101
    assert sum(found["score_hist"]["positive"]) == 20 and sum(found["score_hist"]["negative"]) == 80
    assert answer["overview"]["tasks"] == {"binary": 1} and answer["overview"]["rows"][0]["key"] == "auc"


def test_overrides_change_the_threshold_and_the_task_and_refuse_wrong_values(tmp_path):
    truth, values = anomaly_table(tmp_path / "rec")
    given = stats.prediction_stats(tmp_path / "rec", overrides={"pred_score": {"threshold": 0.0}})["pairs"][0]
    found = given["stats"]
    assert found["threshold"] == 0.0 and found["threshold_rule"] == "given"
    assert found["confusion"][0][1] + found["confusion"][1][1] == int((values >= 0.0).sum())
    regression = stats.prediction_stats(tmp_path / "rec", overrides={"pred_score": {"task": "regression"}})
    pair = regression["pairs"][0]
    assert pair["task"] == "regression" and pair["auto"] == "binary" and pair["stats"]["rmse"] is not None
    with pytest.raises(KalfaError):
        stats.prediction_stats(tmp_path / "rec", overrides={"pred_score": {"task": "multiclass"}})
    with pytest.raises(KalfaError):
        stats.prediction_stats(tmp_path / "rec", overrides={"pred_score": {"positive": "7"}})
    with pytest.raises(KalfaError):
        stats.prediction_stats(tmp_path / "rec", overrides={"pred_score": {"score": "is_anomaly"}})


def test_a_label_target_with_probabilities_is_binary_at_one_half(tmp_path):
    generator = numpy.random.default_rng(1)
    truth = generator.choice(["no", "yes"], size=60, p=[0.6, 0.4])
    prob_yes = numpy.clip(numpy.where(truth == "yes", 0.7, 0.3) + generator.normal(0, 0.2, 60), 0, 1)
    pred = numpy.where(prob_yes >= 0.5, "yes", "no")
    frame = pandas.DataFrame({"row": numpy.arange(60), "churned": truth, "pred_net": pred,
                              "raw_net": (pred == "yes").astype(int), "prob_net_no": 1 - prob_yes,
                              "prob_net_yes": prob_yes})
    write_table(tmp_path / "rec", frame, plan_with([("net", "churned")], ["churned"], classes=["no", "yes"]))
    [pair] = stats.prediction_stats(tmp_path / "rec")["pairs"]
    assert pair["task"] == "binary" and pair["tasks"] == ["binary", "multiclass"]
    assert pair["positive"] == "yes" and pair["score"] == "prob_net_yes"
    assert pair["stats"]["threshold"] == 0.5 and pair["stats"]["threshold_rule"] == "0.5"
    assert pair["stats"]["confusion"][1][1] == int(((prob_yes >= 0.5) & (truth == "yes")).sum())
    flipped = stats.prediction_stats(tmp_path / "rec", overrides={"pred_net": {"positive": "no"}})["pairs"][0]
    assert flipped["positive"] == "no" and flipped["score"] == "prob_net_no"
    multi = stats.prediction_stats(tmp_path / "rec", overrides={"pred_net": {"task": "multiclass"}})["pairs"][0]
    assert multi["stats"]["classes"] == ["no", "yes"] and "macro_f1" in multi["stats"]


def test_multiclass_labels_and_integer_codes(tmp_path):
    generator = numpy.random.default_rng(2)
    truth = generator.choice(["a", "b", "c"], size=90)
    pred = numpy.where(generator.random(90) < 0.7, truth, generator.choice(["a", "b", "c"], size=90))
    write_table(tmp_path / "labels", pandas.DataFrame({"row": numpy.arange(90), "grade": truth, "pred_net": pred}),
                plan_with([("net", "grade")], ["grade"]))
    [pair] = stats.prediction_stats(tmp_path / "labels")["pairs"]
    assert pair["task"] == "multiclass" and pair["tasks"] == ["multiclass"]
    found = pair["stats"]
    assert found["classes"] == ["a", "b", "c"] and sum(map(sum, found["confusion"])) == 90
    assert found["accuracy"] == pytest.approx(float((truth == pred).mean()))
    assert all(sum(row) == pytest.approx(1.0) for row in found["confusion_normalized"])
    assert found["macro_f1"] == pytest.approx(numpy.mean([found["per_class"][label]["f1"] for label in "abc"]))
    codes = pandas.DataFrame({"row": numpy.arange(90), "level": generator.integers(0, 4, 90),
                              "pred_net": generator.integers(0, 4, 90)})
    write_table(tmp_path / "codes", codes, plan_with([("net", "level")], ["level"]))
    [pair] = stats.prediction_stats(tmp_path / "codes")["pairs"]
    assert pair["task"] == "multiclass" and pair["tasks"] == ["multiclass", "regression"]
    assert pair["stats"]["classes"] == ["0", "1", "2", "3"]


def test_counts_with_float_predictions_stay_regression(tmp_path):
    generator = numpy.random.default_rng(3)
    defects = generator.poisson(2.0, 80)
    frame = pandas.DataFrame({"row": numpy.arange(80), "defects": defects,
                              "pred_net": defects + generator.normal(0, 0.5, 80)})
    write_table(tmp_path / "rec", frame, plan_with([("net", "defects")], ["defects"]))
    [pair] = stats.prediction_stats(tmp_path / "rec")["pairs"]
    assert pair["task"] == "regression" and pair["tasks"] == ["regression"] and pair["kind"] == "regression"
    assert pair["stats"]["rmse"] > 0 and pair["stats"]["r2"] is not None


def test_outputs_without_a_target_are_an_embedding_or_values(tmp_path):
    generator = numpy.random.default_rng(4)
    frame = pandas.DataFrame(generator.normal(size=(40, 6)), columns=[f"pred_z_{index}" for index in range(6)])
    frame.insert(0, "row", numpy.arange(40))
    frame["pred_energy"] = generator.normal(size=40)
    frame["path"] = [f"items/{index}.txt" for index in range(40)]
    frame["label"] = generator.choice(["u", "v"], 40)
    write_table(tmp_path / "rec", frame, {"format": 1, "nodes": {}, "data": {"fields": {}}})
    answer = stats.prediction_stats(tmp_path / "rec")
    strict(answer)
    pairs = by_pred(answer)
    assert set(pairs) == {"pred_z", "pred_energy"}
    embedding = pairs["pred_z"]
    assert embedding["task"] == "embedding" and embedding["target"] is None and embedding["images"] is None
    assert embedding["columns"] == [f"pred_z_{index}" for index in range(6)]
    pca = embedding["stats"]["pca"]
    assert len(pca["x"]) == len(pca["y"]) == 40 and pca["color_by"] == "label" and len(pca["color"]) == 40
    assert 0 < sum(pca["explained"]) <= 1.0 + 1e-9
    values = pairs["pred_energy"]
    assert values["task"] == "values" and values["stats"]["count"] == 40
    assert answer["overview"]["tasks"] == {"embedding": 1, "values": 1}


def test_multi_target_tables_have_one_pair_per_target_and_an_overview(tmp_path):
    generator = numpy.random.default_rng(5)
    count = 60
    frame = pandas.DataFrame({"row": numpy.arange(count), "yield": generator.normal(10, 2, count),
                              "moisture": generator.normal(0.3, 0.05, count), "flag": generator.integers(0, 2, count),
                              "grade": generator.choice(["a", "b", "c"], count)})
    frame["pred_y_yield"] = frame["yield"] + generator.normal(0, 0.5, count)
    frame["pred_y_moisture"] = frame["moisture"] + generator.normal(0, 0.01, count)
    frame["pred_y_flag"] = numpy.clip(frame["flag"] * 0.6 + generator.random(count) * 0.4, 0, 1)
    frame["pred_y_grade"] = frame["grade"]
    names = ("yield", "moisture", "flag", "grade")
    write_table(tmp_path / "rec", frame, plan_with([(f"y_{name}", name) for name in names], list(names)))
    answer = stats.prediction_stats(tmp_path / "rec")
    strict(answer)
    pairs = by_pred(answer)
    assert {pred: pair["target"] for pred, pair in pairs.items()} == {f"pred_y_{name}": name for name in names}
    assert {pred: pair["task"] for pred, pair in pairs.items()} == {
        "pred_y_yield": "regression", "pred_y_moisture": "regression", "pred_y_flag": "binary",
        "pred_y_grade": "multiclass"}
    overview = answer["overview"]
    assert overview["pairs"] == 4 and overview["tasks"] == {"regression": 2, "binary": 1, "multiclass": 1}
    assert {row["pred"]: row["key"] for row in overview["rows"]} == {
        "pred_y_yield": "r2", "pred_y_moisture": "r2", "pred_y_flag": "auc", "pred_y_grade": "accuracy"}
    grade = next(row for row in overview["rows"] if row["pred"] == "pred_y_grade")
    assert grade["value"] == 1.0 and grade["count"] == count and grade["target"] == "grade"


def pictures_record(root):
    pictures = root / "pictures"
    pictures.mkdir(parents=True)
    paths = []
    for index in range(6):
        path = pictures / f"{index}.png"
        Image.new("RGB", (8, 8), (index * 40, 0, 0)).save(path)
        paths.append(str(path))
    prob_dog = [0.1, 0.4, 0.2, 0.8, 0.6, 0.9]
    frame = pandas.DataFrame({"row": [10, 11, 12, 13, 14, 15], "label": ["cat", "dog", "cat", "dog", "cat", "dog"],
                              "pred_net": ["cat", "cat", "cat", "dog", "dog", "dog"],
                              "prob_net_cat": [1 - value for value in prob_dog], "prob_net_dog": prob_dog,
                              "path": paths})
    directory = write_table(root / "runs" / "rec", frame, plan_with([("net", "label")], ["label"],
                                                                    classes=["cat", "dog"]))
    secret = root / "secret.txt"
    secret.write_text("not an image", encoding="utf-8")
    frame.assign(path=[str(secret)] * 6).to_parquet(directory / "predictions_other.parquet", index=False)
    return Board(root / "runs")


def test_image_paths_give_a_gallery_and_the_board_serves_only_listed_images(tmp_path):
    board = pictures_record(tmp_path)
    answer = board.predictions("rec")
    strict(answer)
    [pair] = answer["pairs"]
    assert pair["images"] == "path" and pair["task"] == "binary"
    gallery = pair["stats"]["gallery"]
    assert {entry["index"] for entry in gallery[:2]} == {1, 4}
    assert set(gallery[0]) >= {"index", "row", "truth", "pred", "score", "error"}
    body, kind = board.image("rec", 1)
    assert kind == "image/png" and body.startswith(b"\x89PNG")
    with pytest.raises(KalfaError):
        board.image("rec", 99)
    with pytest.raises(KalfaError):
        board.image("rec", 0, name="predictions_other.parquet")
    assert board.predictions("rec", name="predictions_other.parquet")["pairs"][0]["images"] is None
    assert board.predictions("rec", pair="pred_net", threshold=0.3)["pairs"][0]["stats"]["threshold"] == 0.3
    with pytest.raises(KalfaError):
        board.predictions("rec", pair="pred_net", task="regression")


def test_the_server_serves_images_and_task_overrides(tmp_path):
    pictures_record(tmp_path)
    httpd = make_server(tmp_path / "runs", host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{base}/api/record/rec/image?index=1", timeout=10) as response:
            assert response.headers.get("Content-Type", "").startswith("image/png")
            assert response.read().startswith(b"\x89PNG")
        query = "pair=pred_net&task=multiclass"
        with urllib.request.urlopen(f"{base}/api/record/rec/predictions?{query}", timeout=10) as response:
            pair = json.loads(response.read())["pairs"][0]
            assert pair["task"] == "multiclass" and pair["stats"]["classes"] == ["cat", "dog"]
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_the_smaller_label_as_positive_turns_a_numeric_score_around(tmp_path):
    anomaly_table(tmp_path / "rec")
    default = stats.prediction_stats(tmp_path / "rec")["pairs"][0]["stats"]
    flipped = stats.prediction_stats(tmp_path / "rec", overrides={"pred_score": {"positive": "0"}})["pairs"][0]
    assert flipped["positive"] == "0" and flipped["score"] == "-pred_score"
    assert flipped["stats"]["roc"]["auc"] == pytest.approx(default["roc"]["auc"])
    assert flipped["stats"]["threshold_rule"] == "best_f1"
    generator = numpy.random.default_rng(6)
    flag = generator.integers(0, 2, 50)
    frame = pandas.DataFrame({"row": numpy.arange(50), "flag": flag,
                              "pred_net": numpy.clip(flag * 0.6 + generator.random(50) * 0.4, 0, 1)})
    write_table(tmp_path / "unit", frame, plan_with([("net", "flag")], ["flag"]))
    turned = stats.prediction_stats(tmp_path / "unit", overrides={"pred_net": {"positive": "0"}})["pairs"][0]
    assert turned["score"] == "1 - pred_net" and turned["stats"]["threshold"] == 0.5
    straight = stats.prediction_stats(tmp_path / "unit")["pairs"][0]
    assert turned["stats"]["roc"]["auc"] == pytest.approx(straight["stats"]["roc"]["auc"])
