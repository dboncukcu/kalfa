from collections import Counter
from pathlib import Path

import pytest

import kalfa
from kalfa import cli
from kalfa import summary as summarizing
from kalfa.core.errors import KalfaError
from kalfa.records import reader
from records.fixtures import make_record


REAL = Path(__file__).resolve().parents[2] / "runs" / "housing_20261107_183341"


def counting(monkeypatch, name):
    counts = Counter()
    original = getattr(reader, name)

    def wrapped(*args):
        counts[Path(args[-1]).name] += 1
        return original(*args)

    monkeypatch.setattr(reader, name, wrapped)
    return counts


def test_summary_of_parses_each_record_file_once(tmp_path, monkeypatch):
    record = make_record(tmp_path / "alpha")
    lines = counting(monkeypatch, "read_lines")
    jsons = counting(monkeypatch, "read_json")
    reader.summary_of(record)
    assert lines["history.jsonl"] == 1
    assert jsons["manifest.json"] == 1 and jsons["plan.json"] == 1


def test_summarize_by_another_series_parses_history_at_most_twice(tmp_path, monkeypatch):
    record = make_record(tmp_path / "alpha")
    lines = counting(monkeypatch, "read_lines")
    summarizing.summarize([record], by="val/mse")
    assert lines["history.jsonl"] <= 2


def test_progress_status_and_last_agree_with_summary_and_state(tmp_path):
    record = kalfa.open(make_record(tmp_path / "alpha", status="running"))
    assert record.state == "stale" and record.summary["status"] == "stale"
    assert record.progress["status"] == "stale"
    assert record.progress["manifest_status"] == "running"
    assert record.progress["last"] == record.summary["last"]
    assert not {"rules", "minimizes/main", "seconds"} & set(record.progress["last"])


def test_progress_turn_falls_back_to_the_manifest_turn_like_summary(tmp_path):
    record = make_record(tmp_path / "alpha", turns=4)
    (record / "history.jsonl").unlink()
    assert reader.summary_of(record)["turn"] == 4
    assert reader.progress_of(record)["turn"] == 4


def test_best_line_says_no_record_has_the_series_instead_of_naming_one(tmp_path, capsys):
    alpha = make_record(tmp_path / "alpha")
    assert cli.main(["summarize", str(alpha), "--by", "val/nothing"]) == 0
    best = [line for line in capsys.readouterr().out.splitlines() if line.startswith("best:")][0]
    assert best == "best: none of the records has a value for val/nothing"
    found = summarizing.summarize([alpha], by="val/nothing")
    assert found.best is None or found.best["best"] is None


def test_a_param_named_like_a_fixed_column_moves_under_a_param_prefix(tmp_path):
    record = make_record(tmp_path / "alpha", params={"name": "wide", "lr": 0.01})
    found = summarizing.summarize([record])
    assert list(found.table.columns).count("name") == 1 and "param_name" in found.table.columns
    assert found.best["name"] == "alpha" and found.best["param_name"] == "wide"


def test_stats_with_a_bad_where_raises_a_kalfa_error(tmp_path):
    record = kalfa.open(make_record(tmp_path / "alpha"))
    with pytest.raises(KalfaError, match="bad where for predictions.parquet"):
        record.stats(where="nowhere == 1")
    with pytest.raises(KalfaError, match="bad where"):
        record.stats(where="town ==")


def test_csv_keeps_integer_params_when_a_record_lacks_them(tmp_path):
    alpha = make_record(tmp_path / "alpha", params={"seed": 7, "epochs": 100})
    beta = make_record(tmp_path / "beta", params={}, seed=2)
    csv, _ = summarizing.write(summarizing.summarize([alpha, beta]), tmp_path / "out")
    text = Path(csv).read_text(encoding="utf-8")
    assert ",7,100," in text and ",7.0,100.0," not in text


@pytest.mark.skipif(not REAL.is_dir(), reason="the real housing record is absent")
def test_last_columns_follow_the_set_order_not_the_last_history_line():
    history = reader.read_history(REAL)
    assert list(history[0])[2:4] == ["train/mse", "train/huber"]
    assert list(history[-1])[2:4] == ["train/huber", "train/mse"]
    assert list(reader.summary_of(REAL)["last"])[:2] == ["train/huber", "train/mse"]
    table = summarizing.summarize([REAL]).table
    series = [column[len("last_"):] for column in table.columns if column.startswith("last_")][1:]
    sets = [name.split("/")[0] for name in series]
    assert sets == sorted(sets, key=["train", "val", "test", "lr"].index)
    assert series[:5] == sorted(series[:5])
    assert len(table.columns) == 26


@pytest.mark.skipif(not REAL.is_dir(), reason="the real housing record is absent")
def test_text_table_of_one_housing_record_hides_train_and_lr_series_by_default():
    found = summarizing.summarize([REAL])
    narrow = summarizing.text_of(found)
    header = narrow.splitlines()[0].split()
    assert not any(column.startswith(("last_train/", "last_lr/")) for column in header)
    assert "last_val/rmse" in header and "last_test/rmse" in header
    full = summarizing.text_of(found, columns="all")
    assert len(full.splitlines()[0].split()) == 26
    assert max(len(line) for line in full.splitlines()) > 300
    assert max(len(line) for line in narrow.splitlines()) < max(len(line) for line in full.splitlines()) - 80
