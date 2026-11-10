import json
from pathlib import Path

import pandas
import pytest

from kalfa import cli
from kalfa import summary as summarizing
from kalfa.commands import report
from kalfa.core.errors import KalfaError
from kalfa.records import reader
from records.fixtures import make_record, make_sweep


@pytest.fixture
def three(tmp_path):
    alpha = make_record(tmp_path / "alpha", seed=1, turns=3)
    beta = make_record(tmp_path / "beta", seed=2, turns=5, params={"lr": 0.01, "width": 32},
                       started="2026-11-07T11:00:00")
    gamma = make_record(tmp_path / "gamma", seed=3, turns=0, status="running", predictions=False)
    return alpha, beta, gamma


def best_in(directory, series, mode):
    return reader.best_in(reader.read_history(directory), series, mode)


def test_expand_accepts_one_record_or_a_list_and_drops_repeats(three):
    alpha, beta, _ = three
    assert summarizing.expand(str(alpha)) == ([str(alpha)], None, [])
    directories, sweep, points = summarizing.expand([alpha, str(alpha), beta])
    assert directories == [str(alpha), str(beta)] and sweep is None and points == []


def test_expand_walks_a_root_of_records_in_records_under_order(three, tmp_path):
    alpha, beta, gamma = three
    directories, _, _ = summarizing.expand(tmp_path)
    assert directories == [str(beta), str(alpha), str(gamma)]
    assert summarizing.expand([tmp_path, alpha])[0] == [str(beta), str(alpha), str(gamma)]


def test_expand_opens_sweeps_under_a_plain_root_into_their_points(tmp_path):
    alpha = make_record(tmp_path / "alpha", started="2026-11-07T12:00:00")
    _, points = make_sweep(tmp_path / "grid", count=2)
    directories, sweep, found = summarizing.expand(tmp_path)
    assert directories == [str(alpha), *(str(point) for point in points)]
    assert sweep is None and found == []


def test_expand_opens_a_sweep_root_into_its_points(tmp_path):
    root, points = make_sweep(tmp_path / "sweep", count=3)
    directories, sweep, found = summarizing.expand(root)
    assert directories == [str(point) for point in points]
    assert sweep["space"] == {"lr": [0.01, 0.001], "width": [32, 64]} and sweep["points"] == 3
    assert [point["id"] for point in found] == [0, 1, 2]


def test_expand_rejects_unknown_paths_and_empty_roots(three, tmp_path):
    alpha, _, _ = three
    with pytest.raises(KalfaError, match="nowhere"):
        summarizing.expand([alpha, tmp_path / "nowhere"])
    (tmp_path / "empty").mkdir()
    with pytest.raises(KalfaError, match="empty"):
        summarizing.expand(tmp_path / "empty")


def test_default_series_prefers_the_watched_series_then_the_first_val_series(three, tmp_path):
    alpha, _, _ = three
    assert summarizing.default_series([reader.summary_of(alpha)]) == ("val/rmse", "min")
    other = make_record(tmp_path / "other", watch="val/mse", mode="max")
    assert summarizing.default_series([reader.summary_of(other), reader.summary_of(alpha)]) == ("val/mse", "max")
    assert summarizing.default_series([{"best": None, "last": {"train/mse": 1.0, "val/mse": 2.0}}]) == ("val/mse",
                                                                                                        "min")
    assert summarizing.default_series([{"best": None, "last": {}}]) == (None, "min")
    assert summarizing.default_series([]) == (None, "min")


def test_value_of_reuses_best_of_when_the_series_and_mode_match(three):
    alpha, _, gamma = three
    summary = reader.summary_of(alpha)
    best = reader.best_of(alpha)
    assert summarizing.value_of(summary, "val/rmse", "min") == (best["value"], best["turn"])
    assert summarizing.value_of(summary, "val/rmse", "max") == best_in(alpha, "val/rmse", "max")
    assert summarizing.value_of(summary, "val/mse", "min") == best_in(alpha, "val/mse", "min")
    assert summarizing.value_of(summary, "val/rmse", "max")[1] == 1
    assert summarizing.value_of(summary, "nothing/here", "min") == (None, None)
    assert summarizing.value_of(summary, None, "min") == (None, None)
    assert summarizing.value_of(reader.summary_of(gamma), "val/rmse", "min") == (None, None)


def test_table_of_has_the_columns_in_plan_order_and_sorts_by_best(three):
    alpha, beta, gamma = three
    summaries = [reader.summary_of(path) for path in (alpha, beta, gamma)]
    table = summarizing.table_of(summaries, "val/rmse", "min")
    assert list(table.columns) == ["name", "status", "turn", "total", "seconds", "lr", "width", "best", "best_turn",
                                   "last_val/rmse", "last_train/mse", "last_val/mse", "last_test/rmse",
                                   "last_lr/main"]
    assert list(table["name"]) == ["beta", "alpha", "gamma"]
    assert list(table["turn"]) == [5, 3, 0]
    assert table.loc[0, "best"] == best_in(beta, "val/rmse", "min")[0] and table.loc[0, "best_turn"] == 5
    assert list(table["lr"]) == [0.01, 0.001, 0.001] and table.loc[0, "width"] == 32
    assert pandas.isna(table.loc[1, "width"]) and pandas.isna(table.loc[2, "best"])
    assert table.loc[1, "last_val/rmse"] == reader.read_history(alpha)[-1]["val/rmse"]
    assert table.loc[2, "status"] == "stale" and table.loc[2, "seconds"] == 0.0


def test_table_of_sorts_descending_under_max_and_keeps_order_without_a_series(three):
    alpha, beta, gamma = three
    summaries = [reader.summary_of(path) for path in (alpha, beta, gamma)]
    table = summarizing.table_of(summaries, "val/rmse", "max")
    values = [best_in(alpha, "val/rmse", "max")[0], best_in(beta, "val/rmse", "max")[0]]
    assert list(table["name"])[:2] == (["alpha", "beta"] if values[0] >= values[1] else ["beta", "alpha"])
    assert list(table["name"])[2] == "gamma"
    plain = summarizing.table_of(summaries, None, "min")
    assert list(plain["name"]) == ["alpha", "beta", "gamma"]
    assert "last_val/rmse" in plain.columns and plain["best"].isna().all()


def test_summarize_defaults_cuts_and_reports_the_best_row(three):
    alpha, beta, gamma = three
    found = summarizing.summarize([alpha, beta, gamma])
    assert found.by == "val/rmse" and found.mode == "min"
    assert [summary["name"] for summary in found.records] == ["alpha", "beta", "gamma"]
    assert list(found.table["name"]) == ["beta", "alpha", "gamma"]
    assert found.best["name"] == "beta" and isinstance(found.best["best"], float)
    assert found.best["best_turn"] == 5 and found.best["width"] == 32
    assert found.folds is None and found.sweep is None and found.points == []
    short = summarizing.summarize([alpha, beta, gamma], top=1)
    assert list(short.table["name"]) == ["beta"] and short.best["name"] == "beta"
    flipped = summarizing.summarize([alpha, beta], by="val/mse", mode="max")
    expected = max((best_in(path, "val/mse", "max")[0], path.name) for path in (alpha, beta))[1]
    assert flipped.table.loc[0, "name"] == expected and flipped.best["best_turn"] == 1


def test_summarize_fills_sweep_folds_and_handles_no_records(tmp_path):
    root, points = make_sweep(tmp_path / "sweep", count=3)
    found = summarizing.summarize(root)
    assert found.sweep["objective"]["watch"] == "val/rmse" and len(found.points) == 3
    assert set(found.table["name"]) == {"0000", "0001", "0002"} and found.best["name"] in {"0000", "0001", "0002"}
    first = make_record(tmp_path / "f0", params={"lr": 0.001, "fold": 0}, seed=1)
    second = make_record(tmp_path / "f1", params={"lr": 0.001, "fold": 1}, seed=2)
    folded = summarizing.summarize([first, second])
    assert len(folded.folds) == 1 and folded.folds[0]["group"] == {"lr": 0.001}
    assert len(folded.folds[0]["members"]) == 2
    lasts = [member["last"]["val/rmse"] for member in folded.folds[0]["members"]]
    assert folded.folds[0]["mean"]["val/rmse"] == pytest.approx(sum(lasts) / 2)
    assert folded.folds[0]["std"]["val/rmse"] == pytest.approx(abs(lasts[0] - lasts[1]) / 2)
    empty = summarizing.summarize([])
    assert empty.best is None and len(empty.table) == 0 and empty.by is None
    assert summarizing.text_of(empty).splitlines() == ["name  status  turn  total  seconds  best  best_turn",
                                                       "no records"]


def test_text_of_aligns_columns_and_formats_numbers(three):
    alpha, beta, gamma = three
    found = summarizing.summarize([alpha, beta, gamma])
    text = summarizing.text_of(found)
    lines = text.splitlines()
    assert len(lines) == 4 and lines[0].startswith("name")
    column = lines[0].index("status")
    assert [line[column:column + 7].strip() for line in lines[1:]] == ["done", "done", "stale"]
    assert format(found.best["best"], ".4g") in lines[1]
    assert format(found.best["seconds"], ".4g") in lines[1]
    assert "0.01" in lines[1] and "32" in lines[1]
    assert "nan" not in text and "None" not in text
    assert not any(line.startswith("best:") for line in lines)
    assert len(summarizing.text_of(found, top=1).splitlines()) == 2


def test_markdown_of_writes_a_github_table(three):
    alpha, beta, _ = three
    found = summarizing.summarize([alpha, beta])
    lines = summarizing.markdown_of(found).splitlines()
    columns = list(found.table.columns)
    assert lines[0] == "| " + " | ".join(columns) + " |"
    assert lines[1] == "|" + "---|" * len(columns)
    assert len(lines) == 4 and lines[2].startswith("| beta | done | 5 | 10 | 2.5 | 0.01 | 32 | ")
    assert lines[3].startswith("| alpha | done | 3 | 10 | 1.5 | 0.001 |  | ")
    assert found.to_markdown() == summarizing.markdown_of(found)
    assert len(summarizing.markdown_of(found, top=1).splitlines()) == 3


def test_write_puts_csv_and_markdown_into_the_directory(three, tmp_path):
    alpha, beta, _ = three
    found = summarizing.summarize([alpha, beta])
    out = tmp_path / "report" / "deep"
    csv, markdown = summarizing.write(found, out)
    assert csv == str(out / "summary.csv") and markdown == str(out / "summary.md")
    table = pandas.read_csv(csv)
    assert list(table.columns) == list(found.table.columns)
    assert list(table["name"]) == ["beta", "alpha"]
    assert Path(markdown).read_text(encoding="utf-8") == summarizing.markdown_of(found) + "\n"


def test_params_named_like_fixed_columns_move_under_a_param_prefix(tmp_path):
    record = make_record(tmp_path / "alpha", params={"name": "wide", "best": 3, "last_val/rmse": 1, "lr": 0.01})
    found = summarizing.summarize([record])
    columns = list(found.table.columns)
    assert len(set(columns)) == len(columns)
    assert columns[:5] == ["name", "status", "turn", "total", "seconds"]
    assert columns[5:9] == ["param_name", "param_best", "param_last_val/rmse", "lr"]
    assert found.table["name"].tolist() == ["alpha"] and found.table["status"].tolist() == ["done"]
    assert found.table["param_name"].tolist() == ["wide"] and found.table["param_best"].tolist() == [3]
    assert found.best["name"] == "alpha" and found.best["param_name"] == "wide"
    assert isinstance(found.best["best"], float)


def test_text_of_hides_train_and_lr_series_unless_asked(three):
    alpha, beta, gamma = three
    found = summarizing.summarize([alpha, beta, gamma])
    header = summarizing.text_of(found).splitlines()[0].split()
    assert header == ["name", "status", "turn", "total", "seconds", "lr", "width", "best", "best_turn",
                      "last_val/rmse", "last_val/mse", "last_test/rmse"]
    full = summarizing.text_of(found, columns="all").splitlines()[0].split()
    assert full == [str(column) for column in found.table.columns]
    assert "last_train/mse" in full and "last_lr/main" in full
    assert summarizing.text_of(found, columns=["name", "best"]).splitlines()[0].split() == ["name", "best"]
    assert summarizing.text_of(found, columns="name, best").splitlines()[0].split() == ["name", "best"]
    assert len(summarizing.text_of(found, top=1, columns="all").splitlines()) == 2
    with pytest.raises(KalfaError, match="unknown columns: nothing"):
        summarizing.text_of(found, columns=["name", "nothing"])
    by_train = summarizing.summarize([alpha, beta], by="train/mse")
    header = summarizing.text_of(by_train).splitlines()[0].split()
    assert "last_train/mse" in header and "last_lr/main" not in header
    assert summarizing.markdown_of(found).splitlines()[0] == "| " + " | ".join(full) + " |"


def test_expand_rejects_a_sweep_root_without_points(tmp_path):
    root, _ = make_sweep(tmp_path / "sweep", count=0)
    with pytest.raises(KalfaError, match="no records under"):
        summarizing.expand([root])


def test_failed_records_have_no_best_and_sort_last(tmp_path):
    alpha = make_record(tmp_path / "alpha", seed=1)
    broken = make_record(tmp_path / "broken", seed=0, status="failed",
                         failure={"type": "RuntimeError", "message": "boom"})
    found = summarizing.summarize([broken, alpha])
    assert list(found.table["name"]) == ["alpha", "broken"]
    assert pandas.isna(found.table.loc[1, "best"]) and found.table.loc[1, "status"] == "failed"
    assert found.best["name"] == "alpha"
    assert summarizing.summarize([broken]).best is None


def test_summary_best_is_none_when_no_record_has_a_value(tmp_path):
    fresh = make_record(tmp_path / "fresh", status="running", turns=0, predictions=False)
    assert summarizing.summarize([fresh]).best is None
    assert summarizing.summarize([fresh], by="val/nothing").best is None
    assert len(summarizing.summarize([fresh]).table) == 1


def test_last_columns_follow_the_set_order_whatever_the_history_writes(tmp_path):
    alpha = make_record(tmp_path / "alpha")
    history = reader.read_history(alpha)
    shuffled = [{key: line[key] for key in ("lr/main", "test/rmse", "val/mse", "train/mse", "turn", "update",
                                            "val/rmse", "seconds", "rules", "minimizes/main")} for line in history]
    (alpha / "history.jsonl").write_text("".join(json.dumps(line) + "\n" for line in shuffled), encoding="utf-8")
    columns = list(summarizing.summarize([alpha]).table.columns)
    assert columns[columns.index("last_val/rmse"):] == ["last_val/rmse", "last_train/mse", "last_val/mse",
                                                        "last_test/rmse", "last_lr/main"]


def test_seconds_print_without_scientific_notation_and_csv_keeps_integers(tmp_path):
    alpha = make_record(tmp_path / "alpha", params={"seed": 7, "epochs": 100})
    beta = make_record(tmp_path / "beta", params={}, seed=2)
    history = reader.read_history(alpha)
    history[-1]["seconds"] = 36000.5
    (alpha / "history.jsonl").write_text("".join(json.dumps(line) + "\n" for line in history), encoding="utf-8")
    found = summarizing.summarize([alpha, beta])
    text = summarizing.text_of(found)
    assert "36001.5" in text and "e+04" not in text
    csv, _ = summarizing.write(found, tmp_path / "out")
    lines = Path(csv).read_text(encoding="utf-8").splitlines()
    row = dict(zip(lines[0].split(","), [line for line in lines[1:] if line.startswith("alpha")][0].split(",")))
    assert row["seed"] == "7" and row["epochs"] == "100" and row["seconds"] == "36001.5"


def test_fold_groups_print_after_the_best_line_and_go_to_folds_csv_and_the_report(tmp_path, capsys):
    paths = [make_record(tmp_path / f"fold{fold}", params={"lr": 0.001, "fold": fold}, seed=fold) for fold in range(3)]
    found = summarizing.summarize(paths)
    group = found.folds[0]
    shown = [name for name in group["mean"] if not name.startswith(("train/", "lr/"))]
    lines = summarizing.folds_text(found).splitlines()
    assert lines[0] == "folds 0, 1, 2 of lr=0.001: mean and std over 3 records"
    assert lines[1].split() == ["series", "mean", "std"]
    assert sorted(line.split()[0] for line in lines[2:]) == sorted(shown)
    assert "val/rmse" in shown and "train/mse" not in shown
    out = tmp_path / "out"
    assert cli.main(["summarize", *map(str, paths), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert lines[0] in printed and str(out / "folds.csv") in printed
    table = pandas.read_csv(out / "folds.csv")
    assert list(table.columns) == ["lr", "records", "series", "mean", "std"]
    assert set(table["series"]) == set(group["mean"]) and set(table["records"]) == {3}
    rmse = table[table["series"] == "val/rmse"].iloc[0]
    assert rmse["mean"] == pytest.approx(group["mean"]["val/rmse"]) and rmse["std"] == pytest.approx(
        group["std"]["val/rmse"])
    page = report.write_report(paths, out=str(tmp_path / "report.html")).read_text(encoding="utf-8")
    assert '<h2 id="folds">Folds</h2>' in page and lines[0].split(":")[0] in page
    plain = summarizing.summarize([make_record(tmp_path / "plain")])
    assert plain.folds is None and len(summarizing.write(plain, tmp_path / "plain_out")) == 2
