import json

import numpy
import pandas
import pytest
from fixtures import fake_plan, fake_predictions, make_record, write_json

from kalfa.board.answers import jsonable
from kalfa.core.errors import KalfaError
from kalfa.records import stats


def tiny_regression():
    return pandas.DataFrame({"row": [3, 1, 2, 0], "price": [1.0, 2.0, 3.0, 4.0], "pred_net": [1.5, 2.0, 2.0, 6.0],
                             "raw_net": [0.1, 0.2, 0.3, 0.4], "town": ["a", "b", "a", "b"]})


def tiny_classification():
    return pandas.DataFrame({"row": [0, 1, 2, 3, 4, 5],
                             "label": ["no", "no", "no", "yes", "yes", "yes"],
                             "pred_net": ["no", "no", "yes", "yes", "yes", "no"],
                             "raw_net": [0, 0, 1, 1, 1, 0],
                             "prob_net_no": [0.9, 0.8, 0.4, 0.3, 0.2, 0.6],
                             "prob_net_yes": [0.1, 0.2, 0.6, 0.7, 0.8, 0.4]})


def test_read_predictions_missing_file_is_none(tmp_path):
    assert stats.read_predictions(tmp_path) is None
    make_record(tmp_path / "run", predictions=False)
    assert stats.read_predictions(tmp_path / "run") is None
    assert stats.prediction_stats(tmp_path / "run") is None


def test_read_predictions_reads_the_frame_and_filters_with_where(tmp_path):
    directory = make_record(tmp_path / "run")
    frame = stats.read_predictions(directory)
    assert len(frame) == 50 and {"row", "price", "pred_net", "raw_net", "town"} <= set(frame.columns)
    filtered = stats.read_predictions(directory, where="town == 'ank'")
    assert len(filtered) == 25 and set(filtered["town"]) == {"ank"}
    empty = stats.read_predictions(directory, where="price > 1e9")
    assert len(empty) == 0


def test_read_predictions_other_name_and_broken_file(tmp_path):
    directory = make_record(tmp_path / "run")
    fake_predictions(rows=7).to_parquet(directory / "predictions_valid.parquet", index=False)
    assert len(stats.read_predictions(directory, name="predictions_valid.parquet")) == 7
    (directory / "predictions_bad.parquet").write_bytes(b"not a parquet file")
    assert stats.read_predictions(directory, name="predictions_bad.parquet") is None


def test_prediction_pairs_regression_against_fake_plan():
    frame = fake_predictions()
    pairs = stats.prediction_pairs(frame, fake_plan())
    assert pairs == [{"output": "net", "pred": "pred_net", "target": "price", "kind": "regression", "raw": "raw_net",
                      "scores": []}]


def test_prediction_pairs_classification_against_fake_plan():
    classes = ["cold", "mild", "hot"]
    frame = fake_predictions(classes=classes)
    pairs = stats.prediction_pairs(frame, fake_plan(targets=("label",), classes=classes))
    assert pairs == [{"output": "net", "pred": "pred_net", "target": "label", "kind": "classification",
                      "raw": "raw_net", "scores": ["prob_net_cold", "prob_net_mild", "prob_net_hot"]}]


def test_prediction_pairs_without_plan_uses_the_single_target_and_dtype():
    frame = tiny_regression().drop(columns=["town"])
    assert stats.prediction_pairs(frame, None) == [{"output": "net", "pred": "pred_net", "target": "price",
                                                    "kind": "regression", "raw": "raw_net", "scores": []}]
    labelled = tiny_classification().drop(columns=["prob_net_no", "prob_net_yes"])
    assert stats.prediction_pairs(labelled, {})[0]["kind"] == "classification"
    assert stats.prediction_pairs(frame.drop(columns=["price"]), None) == []
    assert stats.prediction_pairs(None, fake_plan()) == []


def test_prediction_pairs_skips_outputs_without_a_target():
    frame = tiny_regression()
    frame["pred_other"] = 1.0
    plan = fake_plan(targets=("price", "size"))
    pairs = stats.prediction_pairs(frame, plan)
    assert [pair["output"] for pair in pairs] == ["net"]


def test_prediction_pairs_match_spread_outputs_with_spread_targets_by_index():
    horizon = pandas.DataFrame({"row": [0, 1], "load_0": [1.0, 2.0], "load_1": [3.0, 4.0],
                                "pred_net_0": [1.5, 2.0], "pred_net_1": [3.0, 5.0], "raw_net_1": [0.1, 0.2]})
    pairs = stats.prediction_pairs(horizon, fake_plan(targets=("load",)))
    assert [(pair["pred"], pair["target"], pair["raw"]) for pair in pairs] == [("pred_net_0", "load_0", None),
                                                                              ("pred_net_1", "load_1", "raw_net_1")]
    assert {pair["kind"] for pair in pairs} == {"regression"}
    scalar = horizon.drop(columns=["load_0", "load_1"]).assign(load=[1.0, 2.0])
    assert stats.prediction_pairs(scalar, fake_plan(targets=("load",))) == []


def test_prediction_pairs_fall_back_only_to_a_declared_target():
    frame = pandas.DataFrame({"row": [0, 1], "path": ["a.png", "b.png"], "pred_score": [0.1, 0.2],
                              "pred_h_0": [0.3, 0.4], "pred_h_1": [0.5, 0.6]})
    assert stats.prediction_pairs(frame, {}, {"targets": []}) == []
    assert [pair["pred"] for pair in stats.prediction_pairs(frame, {}, None)] == ["pred_score"]
    assert stats.prediction_pairs(frame, {}, {"targets": ["path"]})[0]["target"] == "path"


def test_histogram_counts_and_shared_edges():
    found = stats.histogram([0.0, 0.5, 1.0, 1.0], bins=2)
    assert found == {"edges": [0.0, 0.5, 1.0], "counts": [1, 3]}
    shared = stats.histogram([0.25, 0.75, 5.0], edges=[0.0, 0.5, 1.0])
    assert shared == {"edges": [0.0, 0.5, 1.0], "counts": [1, 1]}
    empty = stats.histogram([], bins=4)
    assert len(empty["edges"]) == 5 and empty["counts"] == [0, 0, 0, 0]
    with_nan = stats.histogram([float("nan"), 1.0, 2.0], bins=1)
    assert with_nan == {"edges": [1.0, 2.0], "counts": [2]}


def test_regression_stats_by_hand():
    found = stats.regression_stats(tiny_regression(), "pred_net", "price", bins=4, sample=10, worst=2)
    assert found["count"] == 4
    assert found["rmse"] == pytest.approx(numpy.sqrt((0.25 + 0.0 + 1.0 + 4.0) / 4))
    assert found["mae"] == pytest.approx((0.5 + 0.0 + 1.0 + 2.0) / 4)
    assert found["bias"] == pytest.approx((0.5 + 0.0 - 1.0 + 2.0) / 4)
    assert found["r2"] == pytest.approx(1.0 - 5.25 / 5.0)
    quantiles = found["residual_quantiles"]
    assert list(quantiles) == ["p05", "p25", "p50", "p75", "p95"]
    assert quantiles["p50"] == pytest.approx(0.25)
    assert quantiles["p05"] == pytest.approx(numpy.quantile([0.5, 0.0, -1.0, 2.0], 0.05))
    assert found["pred_hist"]["edges"] == found["truth_hist"]["edges"]
    assert found["pred_hist"]["edges"][0] == pytest.approx(1.0)
    assert found["pred_hist"]["edges"][-1] == pytest.approx(6.0)
    assert found["truth_hist"]["counts"] == [2, 1, 1, 0] and found["pred_hist"]["counts"] == [3, 0, 0, 1]
    assert found["scatter"] == {"row": [0, 1, 2, 3], "truth": [4.0, 2.0, 3.0, 1.0], "pred": [6.0, 2.0, 2.0, 1.5]}
    assert found["worst"] == [{"row": 0, "truth": 4.0, "pred": 6.0, "error": 2.0},
                              {"row": 2, "truth": 3.0, "pred": 2.0, "error": -1.0}]


def test_regression_stats_r2_none_on_constant_truth_and_empty_frame():
    constant = pandas.DataFrame({"row": [0, 1], "price": [2.0, 2.0], "pred_net": [1.0, 3.0]})
    found = stats.regression_stats(constant, "pred_net", "price")
    assert found["r2"] is None and found["rmse"] == pytest.approx(1.0) and found["bias"] == pytest.approx(0.0)
    empty = stats.regression_stats(constant.iloc[:0], "pred_net", "price")
    assert empty["count"] == 0 and empty["rmse"] is None and empty["r2"] is None
    assert empty["scatter"] == {"row": [], "truth": [], "pred": []} and empty["worst"] == []
    assert empty["residual_quantiles"]["p50"] is None


def test_regression_stats_scatter_is_sampled_with_the_seed_and_sorted_by_row():
    frame = fake_predictions(rows=50)
    first = stats.regression_stats(frame, "pred_net", "price", sample=10, seed=3)
    again = stats.regression_stats(frame, "pred_net", "price", sample=10, seed=3)
    other = stats.regression_stats(frame, "pred_net", "price", sample=10, seed=4)
    assert len(first["scatter"]["row"]) == 10 and first["scatter"] == again["scatter"]
    assert first["scatter"]["row"] == sorted(first["scatter"]["row"])
    assert len(set(first["scatter"]["row"])) == 10 and first["scatter"]["row"] != other["scatter"]["row"]
    expected = numpy.random.default_rng(3).choice(50, 10, replace=False)
    assert first["scatter"]["row"] == sorted(int(index) for index in expected)
    assert len(first["worst"]) == 20
    errors = [abs(item["error"]) for item in first["worst"]]
    assert errors == sorted(errors, reverse=True)
    assert found_matches_frame(first["worst"][0], frame)


def found_matches_frame(item, frame):
    line = frame[frame["row"] == item["row"]].iloc[0]
    return item["truth"] == pytest.approx(line["price"]) and item["pred"] == pytest.approx(line["pred_net"])


def test_regression_stats_ignores_non_finite_rows():
    frame = tiny_regression()
    frame.loc[0, "pred_net"] = float("nan")
    found = stats.regression_stats(frame, "pred_net", "price")
    assert found["count"] == 3 and found["mae"] == pytest.approx(1.0)


def test_classification_stats_by_hand():
    found = stats.classification_stats(tiny_classification(), "pred_net", "label", raw="raw_net",
                                       scores=["prob_net_no", "prob_net_yes"])
    assert found["count"] == 6 and found["accuracy"] == pytest.approx(4 / 6)
    assert found["classes"] == ["no", "yes"]
    assert found["confusion"] == [[2, 1], [1, 2]]
    assert found["per_class"]["no"] == {"precision": pytest.approx(2 / 3), "recall": pytest.approx(2 / 3),
                                        "f1": pytest.approx(2 / 3), "support": 3}
    assert found["per_class"]["yes"]["support"] == 3
    truth = numpy.array([0, 0, 0, 1, 1, 1])
    score = numpy.array([0.1, 0.2, 0.6, 0.7, 0.8, 0.4])
    from sklearn.metrics import average_precision_score, roc_auc_score

    assert found["roc"]["auc"] == pytest.approx(roc_auc_score(truth, score))
    assert found["pr"]["ap"] == pytest.approx(average_precision_score(truth, score))
    assert found["roc"]["fpr"][0] == 0.0 and found["roc"]["tpr"][-1] == 1.0
    assert len(found["roc"]["fpr"]) == len(found["roc"]["tpr"])
    assert len(found["pr"]["precision"]) == len(found["pr"]["recall"])
    assert found["pr"]["precision"][-1] == 1.0 and found["pr"]["recall"][-1] == 0.0


def test_classification_stats_zero_division_and_plan_classes():
    frame = pandas.DataFrame({"row": [0, 1, 2], "label": ["a", "a", "b"], "pred_net": ["a", "a", "a"]})
    found = stats.classification_stats(frame, "pred_net", "label", classes=["b", "a", "c"])
    assert found["classes"] == ["b", "a", "c"]
    assert found["confusion"] == [[0, 1, 0], [0, 2, 0], [0, 0, 0]]
    assert found["per_class"]["b"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1}
    assert found["per_class"]["c"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
    assert found["per_class"]["a"]["precision"] == pytest.approx(2 / 3) and found["per_class"]["a"]["recall"] == 1.0
    assert found["roc"] is None and found["pr"] is None
    numeric = pandas.DataFrame({"label": [0, 1, 1], "pred_net": [0, 1, 0]})
    assert stats.classification_stats(numeric, "pred_net", "label")["classes"] == ["0", "1"]


def test_classification_stats_multiclass_has_no_curves_and_single_truth_class_skips_them():
    classes = ["cold", "mild", "hot"]
    frame = fake_predictions(classes=classes)
    found = stats.classification_stats(frame, "pred_net", "label", raw="raw_net",
                                       scores=[f"prob_net_{name}" for name in classes], classes=classes)
    assert found["classes"] == classes and found["roc"] is None and found["pr"] is None
    assert sum(sum(line) for line in found["confusion"]) == 50
    assert found["accuracy"] == pytest.approx((frame["pred_net"] == frame["label"]).mean())
    only_yes = tiny_classification().iloc[3:]
    found = stats.classification_stats(only_yes, "pred_net", "label", scores=["prob_net_no", "prob_net_yes"])
    assert found["classes"] == ["no", "yes"] and found["roc"] is None and found["pr"] is None
    empty = stats.classification_stats(tiny_classification().iloc[:0], "pred_net", "label")
    assert empty["count"] == 0 and empty["accuracy"] is None and empty["classes"] == []


def test_classification_stats_sigmoid_single_score_column():
    frame = tiny_classification().drop(columns=["prob_net_no"])
    found = stats.classification_stats(frame, "pred_net", "label", scores=["prob_net_yes"])
    assert found["roc"]["auc"] == pytest.approx(8 / 9)


def test_prediction_stats_regression_record(tmp_path):
    directory = make_record(tmp_path / "run")
    found = stats.prediction_stats(directory, bins=10, sample=20, worst=5)
    assert found["rows"] == 50 and len(found["pairs"]) == 1
    pair = found["pairs"][0]
    assert {"output", "pred", "target", "kind", "stats"} <= set(pair)
    assert pair["output"] == "net" and pair["target"] == "price" and pair["kind"] == "regression"
    assert pair["stats"]["count"] == 50 and len(pair["stats"]["scatter"]["row"]) == 20
    assert len(pair["stats"]["worst"]) == 5 and len(pair["stats"]["pred_hist"]["counts"]) == 10
    assert pair["stats"]["rmse"] < 10
    filtered = stats.prediction_stats(directory, where="town == 'ist'")
    assert filtered["rows"] == 25 and filtered["pairs"][0]["stats"]["count"] == 25


def test_prediction_stats_classification_record_and_given_plan(tmp_path):
    classes = ["cold", "mild", "hot"]
    directory = make_record(tmp_path / "run", classes=classes)
    found = stats.prediction_stats(directory)
    pair = found["pairs"][0]
    assert pair["kind"] == "classification" and pair["target"] == "label" and pair["task"] == "multiclass"
    assert pair["stats"]["classes"] == classes and set(pair["stats"]["roc"]) == set(classes)
    assert all(0.0 <= curve["auc"] <= 1.0 for curve in pair["stats"]["roc"].values())
    assert set(pair["stats"]["per_class"]) == set(classes)
    given = stats.prediction_stats(directory, plan=fake_plan(targets=("label",), classes=classes))
    assert given["pairs"][0]["stats"]["confusion"] == pair["stats"]["confusion"]


def test_prediction_stats_binary_record_has_curves(tmp_path):
    classes = ["no", "yes"]
    directory = make_record(tmp_path / "run", classes=classes)
    found = stats.prediction_stats(directory)
    pair = found["pairs"][0]
    assert pair["scores"] == ["prob_net_no", "prob_net_yes"]
    assert 0.5 < pair["stats"]["roc"]["auc"] <= 1.0 and 0.0 < pair["stats"]["pr"]["ap"] <= 1.0


def kernel_plan(directory, fields):
    plan = fake_plan(targets=tuple(name for name in fields if name not in ("x0", "x1")))
    plan["data"] = {"inputs": ["x"], "exposes": ["n_features"], "fields": dict(fields)}
    write_json(directory / "plan.json", plan)
    return plan


def fitted_fields(classes=None, target="label", top=None):
    return {"features": [{"pattern": "x*", "columns": ["x0", "x1"], "names": ["x0", "x1"],
                          "chain": ["StandardScaler"]}],
            "targets": [target],
            "target_fields": [{"name": target, "width": 1 if classes is None else len(classes),
                               "chain": ["LabelEncoder"] if classes else ["StandardScaler"], "classes": classes,
                               "class_weights": None}],
            "feature_names": ["x0", "x1"], "classes": classes if top is None else top, "drops": [],
            "spectators": [], "filter": None}


def test_read_fitted_fields_missing_broken_and_read(tmp_path):
    assert stats.read_fitted_fields(tmp_path) is None
    directory = make_record(tmp_path / "run")
    assert stats.read_fitted_fields(directory) is None
    (directory / "fitted").mkdir()
    (directory / "fitted" / "fields.json").write_text("{not json", encoding="utf-8")
    assert stats.read_fitted_fields(directory) is None
    write_json(directory / "fitted" / "fields.json", [1, 2])
    assert stats.read_fitted_fields(directory) is None
    write_json(directory / "fitted" / "fields.json", fitted_fields(["no", "yes"]))
    found = stats.read_fitted_fields(directory)
    assert found["classes"] == ["no", "yes"] and found["target_fields"][0]["name"] == "label"


def test_fitted_classes_prefers_the_target_field_then_the_top_level():
    fields = fitted_fields([0, 1, 2])
    assert stats.fitted_classes(fields, "label") == ["0", "1", "2"]
    assert stats.fitted_classes(fields) == ["0", "1", "2"]
    assert stats.fitted_classes(fields, "other") == ["0", "1", "2"]
    assert stats.fitted_classes(fitted_fields(None), "label") is None
    assert stats.fitted_classes(fitted_fields(None, target="price", top=["a", "b"]), "price") is None
    assert stats.fitted_classes(fitted_fields(None, target="price", top=["a", "b"])) == ["a", "b"]
    assert stats.fitted_classes(None, "label") is None and stats.fitted_classes({}, "label") is None
    assert stats.classes_for(fake_plan(targets=("label",), classes=["x", "y"]), fields, "label") == ["x", "y"]
    assert stats.classes_for({}, fields, "label") == ["0", "1", "2"]
    assert stats.classes_for(None, None, "label") is None


def test_prediction_stats_uses_fitted_classes_when_the_plan_has_none(tmp_path):
    directory = make_record(tmp_path / "run")
    kernel_plan(directory, {"x0": "double", "x1": "double", "label": "int64"})
    frame = pandas.DataFrame({"row": [0, 1, 2, 3, 4], "label": [0, 0, 1, 1, 1], "pred_net": [0, 1, 1, 1, 0]})
    frame.to_parquet(directory / "predictions.parquet", index=False)
    assert stats.prediction_pairs(frame, stats.read_plan(directory))[0]["kind"] == "regression"
    two_valued = stats.prediction_stats(directory)["pairs"][0]
    assert two_valued["kind"] == "classification" and two_valued["task"] == "binary"
    assert two_valued["tasks"] == ["binary", "regression"] and two_valued["positive"] == "1"
    write_json(directory / "fitted" / "fields.json", fitted_fields([0, 1, 2]))
    assert stats.prediction_pairs(frame, stats.read_plan(directory), stats.read_fitted_fields(directory))[0]["kind"] \
        == "classification"
    pair = stats.prediction_stats(directory)["pairs"][0]
    assert pair["kind"] == "classification" and pair["target"] == "label" and pair["scores"] == []
    result = pair["stats"]
    assert result["classes"] == ["0", "1", "2"]
    assert result["confusion"] == [[1, 1, 0], [1, 2, 0], [0, 0, 0]]
    assert result["per_class"]["2"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
    assert result["accuracy"] == pytest.approx(3 / 5) and result["roc"] is None


def test_prediction_stats_fitted_classes_order_binary_curves_and_regression_targets(tmp_path):
    directory = make_record(tmp_path / "run", classes=["no", "yes"])
    kernel_plan(directory, {"x0": "double", "x1": "double", "label": "string"})
    write_json(directory / "fitted" / "fields.json", fitted_fields(["yes", "no"]))
    pair = stats.prediction_stats(directory)["pairs"][0]
    assert pair["stats"]["classes"] == ["yes", "no"]
    frame = stats.read_predictions(directory)
    assert pair["stats"]["roc"]["auc"] == pytest.approx(
        stats.classification_stats(frame, "pred_net", "label", scores=pair["scores"],
                                   classes=["yes", "no"])["roc"]["auc"])
    given = stats.prediction_stats(directory, plan=fake_plan(targets=("label",), classes=["no", "yes"]))
    assert given["pairs"][0]["stats"]["classes"] == ["no", "yes"]
    regression = make_record(tmp_path / "reg")
    kernel_plan(regression, {"x0": "double", "x1": "double", "price": "double"})
    write_json(regression / "fitted" / "fields.json", fitted_fields(None, target="price"))
    pair = stats.prediction_stats(regression)["pairs"][0]
    assert pair["kind"] == "regression" and pair["stats"]["rmse"] < 10


def test_read_predictions_with_a_bad_where_raises_a_kalfa_error(tmp_path):
    record = make_record(tmp_path / "run")
    with pytest.raises(KalfaError, match="bad where for predictions.parquet"):
        stats.read_predictions(record, where="nowhere == 1")
    with pytest.raises(KalfaError, match="bad where"):
        stats.prediction_stats(record, where="town == ")
    assert len(stats.read_predictions(record, where="town == 'ank'")) == 25


def test_prediction_tables_lists_the_parquet_tables_of_a_record(tmp_path):
    record = make_record(tmp_path / "run")
    fake_predictions(rows=5).to_parquet(record / "predictions_best.parquet", index=False)
    fake_predictions(rows=5).to_parquet(record / "predictions_housing.parquet", index=False)
    (record / "predictions.csv").write_text("row\n", encoding="utf-8")
    assert stats.prediction_tables(record) == ["predictions.parquet", "predictions_best.parquet",
                                               "predictions_housing.parquet"]
    assert stats.prediction_tables(make_record(tmp_path / "bare", predictions=False)) == []
    assert stats.prediction_tables(tmp_path / "nowhere") == []


def strict(value):
    return json.dumps(jsonable(value), allow_nan=False)


def task_plan(names, classes=None, spare=False):
    targets = {name: None for name in [*names, *(["spare"] if spare else [])]}
    nodes = {"data": {"slot": "data", "use": "kalfa.std.data.table.Table", "params": {"targets": targets},
                      "keys": {}}}
    for index, name in enumerate(names):
        nodes[f"metrics.m{index}"] = {"slot": "metric", "use": "kalfa.std.metrics.rmse", "params": {},
                                      "keys": {"pred": f"net_{name}" if len(names) > 1 else "net", "target": name}}
    return {"format": 1, "nodes": nodes, "data": {"inputs": ["x"], "fields": {}, "classes": classes}}


def write_task_table(directory, frame, plan):
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(directory / "predictions.parquet", index=False)
    write_json(directory / "plan.json", plan)
    return directory


def test_label_text_writes_integral_numbers_without_decimals():
    assert [stats.label_text(value) for value in (0, 1.0, numpy.float32(2.0), numpy.int64(7), 0.25, "a", True)] == [
        "0", "1", "2", "7", "0.25", "a", "True"]
    assert stats.label_text(float("nan")) is None and stats.label_text(None) is None
    assert stats.labels([0.0, 1.0]) == ["0", "1"] and stats.labels([]) is None


def test_numeric_labels_sort_as_numbers():
    assert stats.ordered_labels(["10", "2", "3", None, "2"]) == ["2", "3", "10"]
    assert stats.ordered_labels(["b", "10", "a", "9"]) == ["9", "10", "a", "b"]
    frame = pandas.DataFrame({"row": range(6), "grade": ["10", "2", "3", "2", "10", "3"],
                              "pred_net": ["10", "2", "2", "2", "3", "3"]})
    [pair] = stats.task_pairs(frame, task_plan(["grade"]))
    assert pair["auto"] == "multiclass" and pair["classes"] == ["2", "3", "10"]
    codes = pandas.DataFrame({"row": range(6), "level": [10, 2, 3, 2, 10, 3], "pred_net": [10, 2, 2, 2, 3, 3]})
    [pair] = stats.task_pairs(codes, task_plan(["level"]))
    assert pair["auto"] == "multiclass" and pair["classes"] == ["2", "3", "10"]


def test_task_detection_follows_the_contract_order():
    generator = numpy.random.default_rng(0)
    size = 40
    frame = pandas.DataFrame({"row": range(size), "flag": generator.integers(0, 2, size).astype(float),
                              "code": generator.integers(0, 5, size), "count": generator.poisson(3.0, size),
                              "price": generator.normal(size=size), "word": generator.choice(["x", "y", "z"], size)})
    frame["pred_net_flag"] = frame["flag"]
    frame["pred_net_code"] = frame["code"]
    frame["pred_net_count"] = frame["count"] + 0.5
    frame["pred_net_price"] = frame["price"]
    frame["pred_net_word"] = frame["word"]
    pairs = {pair["target"]: pair for pair in stats.task_pairs(frame, task_plan(["flag", "code", "count", "price",
                                                                                "word"]))}
    assert {name: (pair["auto"], pair["tasks"]) for name, pair in pairs.items()} == {
        "flag": ("binary", ["binary", "regression"]), "code": ("multiclass", ["multiclass", "regression"]),
        "count": ("regression", ["regression"]), "price": ("regression", ["regression"]),
        "word": ("multiclass", ["multiclass"])}
    assert pairs["flag"]["classes"] == ["0", "1"]
    many = frame.assign(code=numpy.arange(size) % 25, pred_net_code=numpy.arange(size) % 25)
    assert next(pair for pair in stats.task_pairs(many, task_plan(["code"])) if pair["target"] == "code")[
        "auto"] == "regression"


def test_known_classes_make_a_numeric_target_a_classification():
    frame = pandas.DataFrame({"row": range(4), "label": [0, 1, 0, 1], "pred_net": [0, 1, 1, 1]})
    [pair] = stats.task_pairs(frame, task_plan(["label"], classes=["0", "1", "2"]))
    assert pair["auto"] == "multiclass" and pair["tasks"] == ["multiclass"] and pair["classes"] == ["0", "1", "2"]
    plain = {"targets": ["label", "price"],
             "target_fields": [{"name": "label", "classes": ["a", "b"]}, {"name": "price", "classes": None}]}
    assert stats.classes_for({"data": {"classes": ["a", "b"]}}, plain, "price") is None
    assert stats.classes_for({"data": {"classes": ["a", "b"]}}, plain, "label") == ["a", "b"]


def test_nan_rows_are_left_out_of_every_task(tmp_path):
    frame = pandas.DataFrame({"row": range(8), "flag": [0, 1, 0, 1, numpy.nan, 1, 0, 1],
                              "pred_net": [0.1, 0.9, numpy.nan, 0.7, 0.2, 0.4, 0.3, 0.8],
                              "pred_e_0": [1.0, 2.0, numpy.nan, 1.5, 0.5, 2.5, 1.0, 0.0],
                              "pred_e_1": [0.0, 1.0, 1.0, numpy.nan, 1.5, 0.5, 2.0, 1.0],
                              "pred_energy": [1.0, numpy.nan, 2.0, 3.0, numpy.nan, 4.0, 5.0, 6.0]})
    write_task_table(tmp_path / "rec", frame, task_plan(["flag"], spare=True))
    answer = stats.prediction_stats(tmp_path / "rec")
    strict(answer)
    pairs = {pair["pred"]: pair for pair in answer["pairs"]}
    binary = pairs["pred_net"]["stats"]
    assert pairs["pred_net"]["task"] == "binary" and pairs["pred_net"]["score"] == "pred_net"
    assert binary["count"] == 6 and binary["threshold_rule"] == "0.5"
    assert binary["confusion"] == [[2, 0], [1, 3]]
    assert pairs["pred_e"]["task"] == "embedding" and pairs["pred_e"]["stats"]["count"] == 6
    assert pairs["pred_energy"]["stats"]["count"] == 6
    assert pairs["pred_energy"]["stats"]["mean"] == pytest.approx(3.5)
    regression = stats.prediction_stats(tmp_path / "rec", overrides={"pred_net": {"task": "regression"}})
    assert regression["pairs"][0]["stats"]["count"] == 6


def test_a_single_class_keeps_the_rates_defined(tmp_path):
    frame = pandas.DataFrame({"row": range(5), "label": ["no"] * 5, "pred_net": ["no", "yes", "no", "no", "yes"],
                              "prob_net_no": [0.9, 0.3, 0.8, 0.7, 0.4], "prob_net_yes": [0.1, 0.7, 0.2, 0.3, 0.6]})
    write_task_table(tmp_path / "rec", frame, task_plan(["label"], classes=["no", "yes"]))
    [pair] = stats.prediction_stats(tmp_path / "rec")["pairs"]
    found = pair["stats"]
    assert pair["task"] == "binary" and found["confusion"] == [[3, 2], [0, 0]]
    assert found["recall"] == 0.0 and found["specificity"] == pytest.approx(0.6)
    assert found["balanced_accuracy"] == pytest.approx(0.6) and found["roc"] is None and found["pr"] is None
    assert sum(found["score_hist"]["negative"]) == 5 and sum(found["score_hist"]["positive"]) == 0
    overview = stats.prediction_stats(tmp_path / "rec")["overview"]["rows"][0]
    assert overview["key"] == "f1" and overview["value"] == 0.0
    alone = frame.assign(pred_net="no").drop(columns=["prob_net_no", "prob_net_yes"])
    write_task_table(tmp_path / "alone", alone, task_plan(["label"]))
    [pair] = stats.prediction_stats(tmp_path / "alone")["pairs"]
    assert pair["task"] == "multiclass" and pair["stats"]["classes"] == ["no"]
    assert pair["stats"]["confusion"] == [[5]] and pair["stats"]["accuracy"] == 1.0 and pair["stats"]["roc"] is None


def test_all_predictions_equal_as_in_the_images_example(tmp_path):
    generator = numpy.random.default_rng(1)
    truth = generator.choice(["cat", "dog", "owl"], 12)
    probabilities = numpy.tile([0.387, 0.367, 0.246], (12, 1)) + generator.normal(0, 0.001, (12, 3))
    frame = pandas.DataFrame({"row": range(12), "label": truth, "pred_net": ["cat"] * 12,
                              "prob_net_cat": probabilities[:, 0], "prob_net_dog": probabilities[:, 1],
                              "prob_net_owl": probabilities[:, 2]})
    write_task_table(tmp_path / "pets", frame, task_plan(["label"], classes=["cat", "dog", "owl"]))
    answer = stats.prediction_stats(tmp_path / "pets")
    strict(answer)
    found = answer["pairs"][0]["stats"]
    cats = int((truth == "cat").sum())
    assert answer["pairs"][0]["task"] == "multiclass" and found["accuracy"] == pytest.approx(cats / 12)
    assert found["per_class"]["dog"]["precision"] == 0.0 and found["per_class"]["dog"]["f1"] == 0.0
    assert found["confusion_normalized"][1] == [1.0, 0.0, 0.0] and found["macro_f1"] is not None
    flat = pandas.DataFrame({"row": range(5), "price": [1.5, 2.5, 3.5, 4.5, 5.5], "pred_net": [3.5] * 5,
                             "pred_z_0": [0.5] * 5, "pred_z_1": [0.5] * 5, "pred_energy": [2.0] * 5})
    write_task_table(tmp_path / "flat", flat, task_plan(["price"], spare=True))
    answer = stats.prediction_stats(tmp_path / "flat")
    strict(answer)
    pairs = {pair["pred"]: pair for pair in answer["pairs"]}
    assert pairs["pred_net"]["stats"]["r2"] == pytest.approx(-0.0) and len(pairs["pred_net"]["stats"]["worst"]) == 5
    assert pairs["pred_z"]["stats"]["pca"]["explained"] == [0.0, 0.0]
    assert pairs["pred_energy"]["stats"]["std"] == 0.0
    flat_score = pandas.DataFrame({"row": range(4), "flag": [0, 1, 0, 1], "pred_net": [3.0, 3.0, 3.0, 3.0]})
    write_task_table(tmp_path / "flat_score", flat_score, task_plan(["flag"]))
    [pair] = stats.prediction_stats(tmp_path / "flat_score")["pairs"]
    assert pair["task"] == "binary" and pair["score"] is None and pair["stats"]["roc"] is None


def score_table(directory, size=200, seed=0):
    generator = numpy.random.default_rng(seed)
    truth = (generator.random(size) < 0.3).astype(int)
    score = numpy.where(truth == 1, generator.normal(2.0, 1.0, size), generator.normal(0.0, 1.0, size))
    town = numpy.where(numpy.arange(size) % 2 == 0, "ank", "ist")
    frame = pandas.DataFrame({"row": numpy.arange(size), "hit": truth, "pred_net": score, "town": town})
    write_task_table(directory, frame, task_plan(["hit"]))
    return frame


def test_where_filters_keep_the_task_the_threshold_and_the_unfiltered_index(tmp_path):
    frame = score_table(tmp_path / "rec")
    whole = stats.prediction_stats(tmp_path / "rec")["pairs"][0]["stats"]
    only = stats.prediction_stats(tmp_path / "rec", where="hit == 1")
    [pair] = only["pairs"]
    assert only["rows"] == int(frame["hit"].sum()) and pair["task"] == "binary"
    assert pair["stats"]["threshold"] == whole["threshold"] and pair["stats"]["threshold_rule"] == "best_f1"
    assert pair["stats"]["confusion"][0] == [0, 0] and pair["stats"]["roc"] is None
    half = stats.prediction_stats(tmp_path / "rec", where="town == 'ist'")
    assert half["rows"] == 100 and half["pairs"][0]["stats"]["count"] == 100
    with pytest.raises(KalfaError, match="bad where"):
        stats.prediction_stats(tmp_path / "rec", where="town ==")
    picture_dir = tmp_path / "pictures"
    pictures = frame.assign(path=[f"/data/{index}.png" for index in range(len(frame))])
    write_task_table(picture_dir, pictures, task_plan(["hit"]))
    gallery = stats.prediction_stats(picture_dir, where="town == 'ist'")["pairs"][0]["stats"]["gallery"]
    assert len(gallery) == 24 and all(entry["index"] % 2 == 1 for entry in gallery)
    assert all(pictures["row"].iloc[entry["index"]] == entry["row"] for entry in gallery)


def test_the_threshold_sweep_holds_at_most_101_points(tmp_path):
    score_table(tmp_path / "wide", size=500)
    found = stats.prediction_stats(tmp_path / "wide")["pairs"][0]["stats"]
    sweep = found["sweep"]
    assert len(sweep["threshold"]) == 101 and sweep["threshold"] == sorted(sweep["threshold"])
    assert len(sweep["precision"]) == len(sweep["recall"]) == len(sweep["f1"]) == 101
    assert found["f1"] == pytest.approx(max(sweep["f1"]), abs=0.05) and sweep["recall"][0] == 1.0
    unit = pandas.DataFrame({"row": range(6), "hit": [0, 0, 1, 1, 0, 1],
                             "pred_net": [0.1, 0.4, 0.35, 0.8, 0.2, 0.9]})
    write_task_table(tmp_path / "unit", unit, task_plan(["hit"]))
    found = stats.prediction_stats(tmp_path / "unit")["pairs"][0]["stats"]
    assert found["sweep"]["threshold"] == pytest.approx(list(numpy.linspace(0, 1, 101)))
    assert found["threshold"] == 0.5 and found["confusion"] == [[3, 0], [1, 2]]
    assert found["score_hist"]["edges"][0] == 0.0 and found["score_hist"]["edges"][-1] == 1.0


def test_best_threshold_by_hand():
    actual = numpy.array([False, True, False, True, True])
    values = numpy.array([0.0, 3.0, 2.5, 2.0, 1.0])
    assert stats.best_threshold(actual, values) == 1.0
    assert stats.best_threshold(numpy.array([], dtype=bool), numpy.array([])) is None


def embedding_table(directory, size=300, labelled=True):
    generator = numpy.random.default_rng(2)
    frame = pandas.DataFrame(generator.normal(size=(size, 4)) * [3.0, 1.0, 0.5, 0.1],
                             columns=[f"pred_z_{index}" for index in range(4)])
    frame.insert(0, "row", numpy.arange(size) + 100)
    if labelled:
        frame["label"] = generator.choice(["a", "b"], size)
    frame["town"] = numpy.where(numpy.arange(size) % 2 == 0, "ank", "ist")
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(directory / "predictions.parquet", index=False)
    write_json(directory / "plan.json", {"format": 1, "nodes": {}, "data": {"fields": {}}})
    return frame


def test_embedding_points_are_capped_by_sample(tmp_path):
    frame = embedding_table(tmp_path / "rec")
    [pair] = stats.prediction_stats(tmp_path / "rec", sample=50)["pairs"]
    pca = pair["stats"]["pca"]
    assert pair["task"] == "embedding" and pair["stats"]["count"] == 300 and pair["stats"]["dims"] == 4
    assert len(pca["x"]) == len(pca["y"]) == len(pca["row"]) == len(pca["index"]) == len(pca["color"]) == 50
    assert pca["index"] == sorted(pca["index"]) and all(frame["row"].iloc[index] == row
                                                        for index, row in zip(pca["index"], pca["row"]))
    assert pca["explained"][0] > 0.8 and pca["explained"][0] >= pca["explained"][1]
    assert sum(pair["stats"]["norm_hist"]["counts"]) == 300
    filtered = stats.prediction_stats(tmp_path / "rec", where="town == 'ist'", sample=500)["pairs"][0]["stats"]
    assert filtered["count"] == 150 and all(index % 2 == 1 for index in filtered["pca"]["index"])
    embedding_table(tmp_path / "bare", labelled=False)
    assert stats.prediction_stats(tmp_path / "bare")["pairs"][0]["stats"]["pca"]["color_by"] == "town"


def test_values_stats_by_hand():
    frame = pandas.DataFrame({"pred_energy": [1.0, 2.0, 3.0, 4.0, numpy.nan]})
    found = stats.values_stats(frame, "pred_energy", bins=3)
    assert found["count"] == 4 and found["mean"] == 2.5 and found["std"] == pytest.approx(numpy.std([1, 2, 3, 4]))
    assert found["quantiles"]["min"] == 1.0 and found["quantiles"]["max"] == 4.0
    assert found["quantiles"]["p50"] == 2.5 and sum(found["hist"]["counts"]) == 4 and found["labels"] is None
    words = stats.values_stats(pandas.DataFrame({"pred_net": ["yes", "no", "yes", None]}), "pred_net")
    assert words["count"] == 3 and words["labels"] == {"labels": ["yes", "no"], "counts": [2, 1]}
    assert words["hist"] is None and words["mean"] is None


def test_a_binary_pair_without_a_score_reads_the_labels(tmp_path):
    frame = pandas.DataFrame({"row": range(6), "label": ["a", "a", "b", "b", "b", "a"],
                              "pred_net": ["a", "b", "b", "b", "a", "a"]})
    write_task_table(tmp_path / "rec", frame, task_plan(["label"]))
    answer = stats.prediction_stats(tmp_path / "rec")
    [pair] = answer["pairs"]
    found = pair["stats"]
    assert pair["score"] is None and pair["positive"] == "b" and found["threshold"] is None
    assert found["confusion"] == [[2, 1], [1, 2]] and found["roc"] is None and found["sweep"] is None
    assert found["candidates"] == [] and answer["overview"]["rows"][0]["key"] == "f1"
    flipped = stats.prediction_stats(tmp_path / "rec", overrides={"pred_net": {"positive": "a"}})["pairs"][0]
    assert flipped["stats"]["labels"] == ["b", "a"] and flipped["stats"]["confusion"] == [[2, 1], [1, 2]]


def test_overrides_name_known_pairs_and_keys(tmp_path):
    score_table(tmp_path / "rec")
    with pytest.raises(KalfaError, match="no prediction pair"):
        stats.prediction_stats(tmp_path / "rec", overrides={"pred_other": {"task": "binary"}})
    with pytest.raises(KalfaError, match="takes task"):
        stats.prediction_stats(tmp_path / "rec", overrides={"pred_net": {"colour": "red"}})
    with pytest.raises(KalfaError, match="threshold"):
        stats.prediction_stats(tmp_path / "rec", overrides={"pred_net": {"threshold": "high"}})
    given = stats.prediction_stats(tmp_path / "rec", overrides={"pred_net": {"threshold": "1.5", "task": ""}})
    assert given["pairs"][0]["stats"]["threshold"] == 1.5 and given["pairs"][0]["task"] == "binary"


def test_multiclass_roc_is_per_class(tmp_path):
    classes = ["cold", "mild", "hot"]
    directory = make_record(tmp_path / "run", classes=classes)
    [pair] = stats.prediction_stats(directory)["pairs"]
    frame = stats.read_predictions(directory)
    from sklearn.metrics import roc_auc_score

    for label in classes:
        expected = roc_auc_score((frame["label"] == label).astype(int), frame[f"prob_net_{label}"])
        assert pair["stats"]["roc"][label]["auc"] == pytest.approx(expected)
    weighted = sum(item["f1"] * item["support"] for item in pair["stats"]["per_class"].values()) / 50
    assert pair["stats"]["weighted_f1"] == pytest.approx(weighted)


def test_image_column_needs_every_path_to_be_an_image():
    assert stats.image_column(pandas.DataFrame({"path": ["a.PNG", "b.jpg", None]})) == "path"
    assert stats.image_column(pandas.DataFrame({"path": ["a.png", "b.txt"]})) is None
    assert stats.image_column(pandas.DataFrame({"path": [None, None]})) is None
    assert stats.image_column(pandas.DataFrame({"file": ["a.png"]})) is None


def test_regression_gallery_orders_by_the_largest_error(tmp_path):
    frame = pandas.DataFrame({"row": range(30), "price": numpy.arange(30, dtype=float),
                              "pred_net": numpy.arange(30, dtype=float) + numpy.r_[numpy.zeros(29), 9.0],
                              "path": [f"/img/{index}.jpg" for index in range(30)]})
    write_task_table(tmp_path / "rec", frame, task_plan(["price"]))
    [pair] = stats.prediction_stats(tmp_path / "rec")["pairs"]
    gallery = pair["stats"]["gallery"]
    assert pair["images"] == "path" and len(gallery) == 24
    assert gallery[0] == {"index": 29, "row": 29, "truth": 29.0, "pred": 38.0, "score": None, "error": 9.0}


def float_label_table(directory, size=80, pictures=False):
    generator = numpy.random.default_rng(3)
    truth = (generator.random(size) < 0.5).astype(float)
    prob = numpy.clip(numpy.where(truth == 1.0, 0.72, 0.28) + generator.normal(0, 0.15, size), 0.0, 1.0)
    frame = pandas.DataFrame({"row": numpy.arange(size), "y": truth, "pred_net": (prob >= 0.5).astype(float),
                              "raw_net": (prob >= 0.5).astype("int64"), "prob_net_0.0": 1.0 - prob,
                              "prob_net_1.0": prob})
    if pictures:
        frame["path"] = [f"/img/{index}.png" for index in range(size)]
    write_task_table(directory, frame, task_plan(["y"], classes=[0.0, 1.0]))
    return truth, prob, frame


def test_float_class_labels_find_their_probability_columns(tmp_path):
    from sklearn.metrics import roc_auc_score

    truth, prob, frame = float_label_table(tmp_path / "rec", pictures=True)
    [pair] = stats.prediction_stats(tmp_path / "rec")["pairs"]
    found = pair["stats"]
    assert pair["task"] == "binary" and pair["positive"] == "1" and found["labels"] == ["0", "1"]
    assert pair["score"] == "prob_net_1.0" and found["candidates"] == ["prob_net_1.0", "1 - prob_net_0.0"]
    assert found["roc"]["auc"] == pytest.approx(roc_auc_score(truth, prob)) and found["roc"]["auc"] > 0.9
    assert found["accuracy"] == pytest.approx(float((frame["pred_net"] == frame["y"]).mean()))
    assert all(entry["score"] is not None for entry in found["gallery"])
    [multi] = stats.prediction_stats(tmp_path / "rec", overrides={"pred_net": {"task": "multiclass"}})["pairs"]
    assert set(multi["stats"]["roc"]) == {"0", "1"}
    assert multi["stats"]["roc"]["1"]["auc"] == pytest.approx(roc_auc_score(truth, prob))
    assert all(entry["score"] is not None for entry in multi["stats"]["gallery"])
    plain = stats.classification_stats(frame, "pred_net", "y", raw="raw_net",
                                       scores=["prob_net_0.0", "prob_net_1.0"], classes=["0", "1"])
    assert plain["classes"] == ["0", "1"] and plain["roc"]["auc"] == pytest.approx(roc_auc_score(truth, prob))
    assert stats.classification_stats(frame, "pred_net", "y", classes=[0.0, 1.0])["classes"] == ["0", "1"]
    assert stats.positive_column(["prob_net_0.0", "prob_net_1.0"], "net", ["1", "0"]) == "prob_net_0.0"


def test_a_probability_suffix_matches_a_label_by_its_text_first():
    columns = ["prob_net_01", "prob_net_1.0", "prob_net_x"]
    assert stats.score_labels(columns, "net", ["01", "1"]) == {"prob_net_01": "01", "prob_net_1.0": "1",
                                                               "prob_net_x": "x"}
    assert stats.label_columns(columns, "net", ["01", "1"])["1"] == "prob_net_1.0"


def test_a_positive_class_without_its_own_probability_scores_with_the_complement(tmp_path):
    from sklearn.metrics import roc_auc_score

    generator = numpy.random.default_rng(5)
    truth = generator.choice(["no", "yes"], 200)
    prob_yes = numpy.clip(numpy.where(truth == "yes", 0.75, 0.25) + generator.normal(0, 0.15, 200), 0.0, 1.0)
    frame = pandas.DataFrame({"row": range(200), "churned": truth,
                              "pred_net": numpy.where(prob_yes >= 0.5, "yes", "no"),
                              "raw_net": (prob_yes >= 0.5).astype("int64"), "prob_net_yes": prob_yes})
    write_task_table(tmp_path / "rec", frame, task_plan(["churned"], classes=["no", "yes"]))
    [plain] = stats.prediction_stats(tmp_path / "rec")["pairs"]
    [flipped] = stats.prediction_stats(tmp_path / "rec", overrides={"pred_net": {"positive": "no"}})["pairs"]
    assert plain["score"] == "prob_net_yes" and plain["stats"]["candidates"] == ["prob_net_yes"]
    assert flipped["score"] == "1 - prob_net_yes" and flipped["stats"]["candidates"] == ["1 - prob_net_yes"]
    expected = roc_auc_score(truth == "no", 1.0 - prob_yes)
    assert expected > 0.9 and flipped["stats"]["roc"]["auc"] == pytest.approx(expected)
    assert flipped["stats"]["threshold_rule"] == "0.5" and flipped["stats"]["labels"] == ["yes", "no"]
    assert flipped["stats"]["accuracy"] == pytest.approx(plain["stats"]["accuracy"])
    (tn, fp), (fn, tp) = plain["stats"]["confusion"]
    assert flipped["stats"]["confusion"] == [[tp, fn], [fp, tn]]
    with pytest.raises(KalfaError, match="score must be one of"):
        stats.prediction_stats(tmp_path / "rec", overrides={"pred_net": {"positive": "no", "score": "prob_net_yes"}})
    given = stats.prediction_stats(tmp_path / "rec", overrides={"pred_net": {"score": "1 - prob_net_yes",
                                                                             "positive": "no", "threshold": 0.3}})
    assert given["pairs"][0]["stats"]["threshold"] == 0.3 and given["pairs"][0]["score"] == "1 - prob_net_yes"


def bins_of(directory, pred="pred_net", bins=40, **wanted):
    answer = stats.prediction_stats(directory, bins=bins, overrides={pred: {"bins": wanted}})
    return next(pair for pair in answer["pairs"] if pair["pred"] == pred)["stats"]


def values_table(directory):
    frame = pandas.DataFrame({"row": range(40), "pred_energy": numpy.linspace(0.0, 1.0, 40)})
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(directory / "predictions.parquet", index=False)
    write_json(directory / "plan.json", {"format": 1, "nodes": {}, "data": {"fields": {}}})


def test_each_task_counts_its_own_histograms_with_their_own_bins(tmp_path):
    directory = make_record(tmp_path / "run")
    found = bins_of(directory, residual_hist=3, truth_hist=6)
    assert len(found["residual_hist"]["counts"]) == 3 and len(found["truth_hist"]["counts"]) == 6
    assert found["pred_hist"]["edges"] == found["truth_hist"]["edges"] and sum(found["residual_hist"]["counts"]) == 50
    left = bins_of(directory, bins=10, residual_hist=3)
    assert len(left["truth_hist"]["counts"]) == len(left["pred_hist"]["counts"]) == 10
    assert stats.HISTOGRAMS["regression"] == ("residual_hist", "truth_hist") and stats.MAX_BINS == 1000
    score_table(tmp_path / "score")
    score = bins_of(tmp_path / "score", score_hist=7)["score_hist"]
    assert len(score["edges"]) == 8 and len(score["negative"]) == len(score["positive"]) == 7
    assert sum(score["negative"]) + sum(score["positive"]) == 200
    embedding_table(tmp_path / "embedding")
    assert len(bins_of(tmp_path / "embedding", pred="pred_z", norm_hist=5)["norm_hist"]["counts"]) == 5
    values_table(tmp_path / "values")
    hist = bins_of(tmp_path / "values", pred="pred_energy", hist=4)["hist"]
    assert hist["counts"] == [10, 10, 10, 10]


def test_a_histogram_the_task_does_not_draw_is_refused(tmp_path):
    directory = make_record(tmp_path / "run")
    for name in ("score_hist", "pred_hist", "norm_hist", "hist"):
        with pytest.raises(KalfaError, match=f"no histogram {name}; its histograms are residual_hist, truth_hist"):
            bins_of(directory, **{name: 5})
    score_table(tmp_path / "score")
    with pytest.raises(KalfaError, match="read as binary has no histogram residual_hist"):
        bins_of(tmp_path / "score", residual_hist=5)
    classes = make_record(tmp_path / "classes", classes=["cold", "mild", "hot"])
    with pytest.raises(KalfaError, match="its histograms are none"):
        bins_of(classes, hist=5)
    for wrong in (0, -3, "many", 2.5, True, None, "4.5"):
        with pytest.raises(KalfaError, match="bins of residual_hist for pred_net"):
            bins_of(directory, residual_hist=wrong)
    with pytest.raises(KalfaError, match="must map histogram names to counts"):
        stats.prediction_stats(directory, overrides={"pred_net": {"bins": 12}})


def test_bins_above_the_limit_come_down_to_it_and_text_counts_are_read(tmp_path):
    directory = make_record(tmp_path / "run")
    found = bins_of(directory, residual_hist=5000, truth_hist=numpy.int64(1000))
    assert len(found["residual_hist"]["counts"]) == 1000 and len(found["truth_hist"]["counts"]) == 1000
    assert len(bins_of(directory, residual_hist=" 12 ")["residual_hist"]["counts"]) == 12
    plain = stats.prediction_stats(directory, overrides={"pred_net": {"bins": {}}})["pairs"][0]["stats"]
    assert len(plain["residual_hist"]["counts"]) == 40
    unset = stats.prediction_stats(directory, overrides={"pred_net": {"bins": None}})["pairs"][0]["stats"]
    assert len(unset["truth_hist"]["counts"]) == 40


def test_bins_follow_the_task_chosen_beside_them(tmp_path):
    score_table(tmp_path / "rec")
    turned = stats.prediction_stats(tmp_path / "rec", overrides={
        "pred_net": {"task": "regression", "bins": {"residual_hist": 4}}})["pairs"][0]
    assert turned["task"] == "regression" and len(turned["stats"]["residual_hist"]["counts"]) == 4
    assert len(turned["stats"]["truth_hist"]["counts"]) == 40
    with pytest.raises(KalfaError, match="read as regression has no histogram score_hist"):
        stats.prediction_stats(tmp_path / "rec", overrides={"pred_net": {"task": "regression",
                                                                         "bins": {"score_hist": 4}}})
    given = stats.prediction_stats(tmp_path / "rec", overrides={
        "pred_net": {"threshold": 1.5, "positive": "0", "bins": {"score_hist": 6}}})["pairs"][0]
    found = given["stats"]
    assert found["threshold"] == 1.5 and found["threshold_rule"] == "given" and given["positive"] == "0"
    assert len(found["score_hist"]["positive"]) == 6 and found["score"] == "-pred_net"
    plain = stats.prediction_stats(tmp_path / "rec", overrides={"pred_net": {"threshold": 1.5}})["pairs"][0]
    assert plain["stats"]["confusion"] == stats.prediction_stats(tmp_path / "rec", overrides={
        "pred_net": {"threshold": 1.5, "bins": {"score_hist": 6}}})["pairs"][0]["stats"]["confusion"]
