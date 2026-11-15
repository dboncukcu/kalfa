import re
from pathlib import Path

import numpy
import pandas
import pytest

import kalfa
from helpers import housing_frame
from kalfa import summary as summarizing
from kalfa.commands import report
from kalfa.commands.report import sections
from kalfa.core.errors import KalfaError
from records.fixtures import make_record, make_sweep, write_lines


PNG = "data:image/png;base64,"


def housing_config(home):
    return {
        "seed": 3,
        "device": "cpu",
        "data": {"use": "table", "file": str(home / "housing.parquet"), "split": [0.7, 0.15, 0.15],
                 "features": {"x*": "standard_scaler"}, "targets": {"price": "standard_scaler"}, "batch": 32},
        "models": {"net": {"use": "mlp", "hidden": [8], "out": 1}},
        "losses": {"mse": "mse"},
        "metrics": {"rmse": "rmse"},
        "optimizers": {"main": {"use": "adam", "lr": 0.01, "over": "@models.net", "minimizes": "@losses.mse"}},
        "train": {"epochs": 2, "checkpoint": {"watch": "val/rmse"}, "report": "best"},
    }


def sources_in(text):
    return re.findall(r'src="([^"]*)"', text)


def captions(text):
    return re.findall(r"<figcaption>([^<]*)</figcaption>", text)


def headings(text, level):
    return re.findall(rf"<h{level}>([^<]*)</h{level}>", text)


def assert_self_contained(text):
    assert text.startswith("<!doctype html>")
    assert text.count("<html") == 1 and text.count("</html>") == 1 and text.rstrip().endswith("</html>")
    assert text.count("<style>") == 1 and "<link" not in text
    assert "<script" not in text and "http://" not in text and "https://" not in text
    assert sources_in(text) and all(source.startswith(PNG) for source in sources_in(text))


@pytest.fixture
def pair(tmp_path):
    alpha = make_record(tmp_path / "alpha", seed=1, turns=4)
    beta = make_record(tmp_path / "beta", seed=2, turns=3, params={"lr": 0.01, "width": 32})
    return alpha, beta


def test_write_report_defaults_inside_the_first_record(pair):
    alpha, beta = pair
    path = report.write_report([str(alpha), str(beta)])
    assert isinstance(path, Path) and path == alpha / "report.html"
    text = path.read_text(encoding="utf-8")
    assert_self_contained(text)
    assert "alpha" in text and "beta" in text and "width" in text
    for series in ("val/rmse", "val/mse", "train/mse"):
        assert f"{series} over the epochs of 2 records" in captions(text)
    assert set(headings(text, 3)) == {"alpha", "beta"}
    assert re.search(r"best: (alpha|beta), val/rmse ", text)


def test_write_report_takes_one_path_and_an_out_file_or_directory(pair, tmp_path):
    alpha, _ = pair
    out = tmp_path / "deep" / "one.html"
    assert report.write_report(str(alpha), out=str(out)) == out and out.is_file()
    folder = tmp_path / "reports"
    folder.mkdir()
    assert report.write_report(alpha, out=folder) == folder / "report.html"
    assert (folder / "report.html").is_file()
    assert not list(tmp_path.glob("deep/.*.tmp"))


def test_regression_details_carry_the_scatter_and_histograms(pair, tmp_path):
    alpha, _ = pair
    text = report.write_report(alpha, out=tmp_path / "r.html").read_text(encoding="utf-8")
    assert "price against pred_net, regression" in text
    assert "pred_net against price over 50 rows, and their histograms" in captions(text)
    assert "<th>rmse</th>" in text and "<th>r2</th>" in text
    assert "lr" in text and "last values" in text and "best val/rmse " in text


def test_classification_details_carry_the_confusion_matrix_and_roc(tmp_path):
    record = make_record(tmp_path / "clf", classes=["no", "yes"], watch="val/rmse")
    text = report.write_report(record).read_text(encoding="utf-8")
    assert_self_contained(text)
    assert "label against pred_net, classification" in text
    assert "confusion matrix of pred_net against label, and the roc curve" in captions(text)
    assert "<th>accuracy</th>" in text and "<th>roc auc</th>" in text and "<th>precision</th>" in text


def test_multiclass_embedding_and_values_pairs_get_their_own_sections(tmp_path):
    record = make_record(tmp_path / "many", classes=["a", "b", "c"])
    frame = pandas.read_parquet(record / "predictions.parquet")
    generator = numpy.random.default_rng(0)
    for index in range(4):
        frame[f"pred_z_{index}"] = generator.normal(size=len(frame))
    frame.to_parquet(record / "predictions.parquet", index=False)
    text = report.write_report(record).read_text(encoding="utf-8")
    assert_self_contained(text)
    assert "label against pred_net, classification (multiclass)" in text
    assert "confusion matrix of pred_net against label, and the roc curves" in captions(text)
    assert "<th>macro f1</th>" in text and text.count("<th>roc auc</th>") == 1
    assert "pred_z, 4 columns, embedding" in text and "None against" not in text
    assert "pca map of pred_z over 50 rows, colored by label, and the length of each vector" in captions(text)
    loose = make_record(tmp_path / "loose")
    pandas.read_parquet(loose / "predictions.parquet").drop(columns=["price"]).to_parquet(
        loose / "predictions.parquet", index=False)
    text = report.write_report(loose).read_text(encoding="utf-8")
    assert "<h5>pred_net, values</h5>" in text and "<th>p50</th>" in text
    assert "distribution of pred_net over 50 rows" in captions(text)


def test_a_binary_section_names_the_score_and_the_threshold(tmp_path):
    record = make_record(tmp_path / "clf", classes=["no", "yes"], watch="val/rmse")
    text = report.write_report(record).read_text(encoding="utf-8")
    assert "label against pred_net, classification (binary)" in text
    assert "prob_net_yes &gt;= 0.5 reads as yes, the threshold 0.5 of a probability" in text
    assert "<th>balanced accuracy</th>" in text


def test_a_record_without_history_or_predictions_still_renders(tmp_path):
    empty = make_record(tmp_path / "empty", turns=0, status="running", predictions=False)
    text = report.write_report(empty).read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>") and "empty" in text
    assert "no train/ or val/ series in any history yet" in text
    assert PNG not in text and "predictions.parquet" not in text


def test_a_failed_record_with_broken_predictions_still_renders(tmp_path):
    broken = make_record(tmp_path / "broken", status="failed", failure={"type": "RuntimeError", "message": "boom"},
                         predictions=False)
    (broken / "predictions.parquet").write_bytes(b"not parquet")
    text = report.write_report(broken).read_text(encoding="utf-8")
    assert "failed: boom" in text and "predictions.parquet could not be read" in text


def test_render_draws_the_named_curves_and_skips_stats(pair):
    alpha, beta = pair
    found = summarizing.summarize([alpha, beta])
    text = report.render(found, curves=["val/rmse"], stats=False)
    assert captions(text) == ["val/rmse over the epochs of 2 records"]
    assert "<h4>predictions</h4>" not in text
    assert PNG not in report.render(found, curves=False, stats=False)
    assert len(captions(report.render(found, curves="val/mse, train/mse", stats=False))) == 2


def test_curves_put_val_first_and_stop_at_eight(tmp_path):
    record = make_record(tmp_path / "many", predictions=False)
    lines = []
    for turn in range(1, 4):
        line = {"turn": turn, "train/loss": 1.0 / turn, "test/rmse": 0.5}
        line.update({f"val/s{index:02d}": index + 1.0 / turn for index in range(10)})
        lines.append(line)
    write_lines(record / "history.jsonl", lines)
    text = report.render(summarizing.summarize(record, by="val/s05"), stats=False)
    drawn = [caption.split(" over ")[0] for caption in captions(text)]
    assert drawn == ["val/s05", "val/s00", "val/s01", "val/s02", "val/s03", "val/s04", "val/s06", "val/s07"]
    assert "3 more series not drawn" in text


def test_sweep_report_lists_points_params_and_scatters(tmp_path):
    root, points = make_sweep(tmp_path / "sweep", count=3)
    path = report.write_report(root)
    assert path == root / "report.html"
    text = path.read_text(encoding="utf-8")
    assert_self_contained(text)
    assert '<h2 id="sweep">Sweep</h2>' in text
    for point in ("0000", "0001", "0002"):
        assert f"<td>{point}</td>" in text
    assert "<th>lr</th>" in text and "<th>width</th>" in text and "1 failed" in text and "strategy grid" in text
    assert "val/rmse against lr over 2 points with an objective" in captions(text)
    assert "val/rmse against width over 2 points with an objective" in captions(text)
    assert re.search(r"best point 000[01]: val/rmse ", text)
    assert summarizing.summarize(root).points and str(root) in text


def test_points_limit_the_details(tmp_path):
    root, _ = make_sweep(tmp_path / "sweep", count=3)
    text = report.write_report(root, points="1, 0002", out=tmp_path / "some.html").read_text(encoding="utf-8")
    assert headings(text, 3) == ["0001", "0002"]
    text = report.write_report([root], points=[0], out=tmp_path / "one.html").read_text(encoding="utf-8")
    assert headings(text, 3) == ["0000"]
    with pytest.raises(KalfaError, match="no record or point 7"):
        report.write_report(root, points="7", out=tmp_path / "none.html")
    assert not (tmp_path / "none.html").exists()


def test_details_stop_at_the_limit_unless_points_say_all(tmp_path, monkeypatch):
    for index in range(3):
        make_record(tmp_path / "runs" / f"run{index}", seed=index, predictions=False)
    monkeypatch.setattr(sections, "DETAILS", 2)
    found = summarizing.summarize(tmp_path / "runs")
    text = report.render(found, curves=False)
    assert len(headings(text, 3)) == 2 and "the first 2 of 3 records in the ranking" in text
    assert "--points IDS names the others" in text
    assert len(headings(report.render(found, points="all", curves=False), 3)) == 3


def test_report_needs_records(tmp_path):
    with pytest.raises(KalfaError):
        report.write_report([])
    with pytest.raises(KalfaError):
        report.write_report(tmp_path / "missing")


@pytest.mark.slow
def test_a_trained_record_reports_its_curves_and_predictions(tmp_path):
    housing_frame(160, 0).to_parquet(tmp_path / "housing.parquet", index=False)
    record = Path(kalfa.run(housing_config(tmp_path), record=str(tmp_path / "runs" / "base")).record)
    path = report.write_report([str(record)])
    assert path == record / "report.html"
    text = path.read_text(encoding="utf-8")
    assert_self_contained(text)
    assert "base" in text and "val/rmse" in text
    assert any(caption.startswith("val/rmse over the ") for caption in captions(text))
    assert "price against pred_net, regression" in text


def test_a_param_with_one_value_draws_no_scatter(tmp_path):
    root, _ = make_sweep(tmp_path / "sweep", count=2, space={"lr": [0.01, 0.001], "width": [32]})
    text = report.write_report(root).read_text(encoding="utf-8")
    scatters = [caption for caption in captions(text) if caption.startswith("val/rmse against")]
    assert scatters == ["val/rmse against lr over 2 points with an objective"]
    assert "<th>width</th>" in text
