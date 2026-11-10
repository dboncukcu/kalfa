import json
from pathlib import Path

import pandas
import pytest

from helpers import housing_frame
from kalfa import cli, summary
from kalfa.records.reader import read_lines
from records.fixtures import make_record


CONFIG = """params:
  epochs: 100
  lr: 1.0e-3
  seed: 7
seed: $seed$
device: cpu
data:
  use: table
  file: FILE
  split: [0.7, 0.15, 0.15]
  features: {"x*": standard_scaler}
  targets: {price: standard_scaler}
  batch: 128
models:
  net: {use: mlp, hidden: [32, 32], out: 1}
losses:
  mse: mse
  huber: {use: torch.nn.HuberLoss, delta: 1.0}
metrics:
  rmse: rmse
  mae: mae
optimizers:
  main: {use: adam, lr: $lr$, over: "@models.net", minimizes: "@losses.mse"}
train:
  epochs: $epochs$
  checkpoint: {watch: val/rmse}
  report: best
rules:
  to_huber:
    when: {use: after_epoch, at: 1}
    set: {optimizers.main.minimizes: "@losses.huber"}
plots:
  loss_curve: loss_curve
record: RECORD
"""


@pytest.fixture
def housing(tmp_path):
    path = tmp_path / "housing.parquet"
    housing_frame().to_parquet(path, index=False)
    return path


@pytest.fixture
def config(tmp_path, housing):
    path = tmp_path / "config.yaml"
    text = CONFIG.replace("FILE", str(housing)).replace("RECORD", str(tmp_path / "runs" / "housing"))
    path.write_text(text, encoding="utf-8")
    return path


def test_check_ok(config, capsys):
    assert cli.main(["check", str(config)]) == 0
    out = capsys.readouterr().out
    assert "error" not in out
    assert out.strip() == "ok" or "warning" in out


def test_check_reports_errors(config, capsys):
    broken = config.with_name("broken.yaml")
    broken.write_text(config.read_text(encoding="utf-8").replace("out: 1}", "out: 1, nonsense: 1}"),
                      encoding="utf-8")
    assert cli.main(["check", str(broken)]) == 1
    out = capsys.readouterr().out
    assert "unknown_key" in out and "nonsense" in out


def test_check_prints_the_plan(config, capsys):
    assert cli.main(["check", str(config), "--plan", "--set", "train.epochs=3", "-p", "lr=0.01"]) == 0
    out = capsys.readouterr().out
    plan = json.loads(out[out.index("{"):])
    assert plan["format"] == 1
    assert "models.net" in plan["nodes"]
    assert plan["train"]["epochs"] == 3
    assert plan["nodes"]["optimizers.main"]["params"]["lr"] == 0.01


def test_board_and_tui_are_commands():
    assert {"board", "tui"} <= set(cli.HANDLERS)
    with pytest.raises(SystemExit) as caught:
        cli.main(["board", "--help"])
    assert caught.value.code == 0
def test_no_command_prints_help(capsys):
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert "check" in out and "predict" in out


def test_run_without_config_is_a_usage_error(capsys):
    assert cli.main(["run"]) == 2
    assert "--from" in capsys.readouterr().err


def test_check_of_a_missing_file_is_an_error(tmp_path, capsys):
    assert cli.main(["check", str(tmp_path / "missing.yaml")]) == 1


@pytest.mark.slow
def test_run_predict_and_resume(config, housing, tmp_path, capsys):
    record = tmp_path / "record"
    assert cli.main(["run", str(config), "--set", "train.epochs=2", "--record", str(record)]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == str(record)
    assert "turn 1/2" in captured.err and "turn 2/2" in captured.err
    history = read_lines(record / "history.jsonl")
    assert [line["turn"] for line in history] == [1, 2]
    assert history[1]["minimizes/main"] == "huber"

    assert cli.main(["predict", str(record), "--data", str(housing), "--which", "last"]) == 0
    written = Path(capsys.readouterr().out.strip())
    assert written == record / "predictions_housing.parquet" and written.exists()
    table = pandas.read_parquet(written)
    assert len(table) == 200 and {"pred_net", "price", "row"} <= set(table.columns)

    assert cli.main(["predict", str(record)]) == 0
    assert Path(capsys.readouterr().out.strip()) == record / "predictions_best.parquet"
    assert (record / "predictions.parquet").exists()

    target = tmp_path / "resumed"
    assert cli.main(["run", "--from", str(record), "--record", str(target), "--set", "train.epochs=3",
                     "--quiet"]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == str(target) and "turn" not in captured.err
    assert [line["turn"] for line in read_lines(target / "history.jsonl")] == [1, 2, 3]


def test_check_of_a_record_reads_its_resolved_config(tmp_path, capsys):
    record = tmp_path / "record"
    record.mkdir()
    (record / "resolved.yaml").write_text("models:\n  net: {use: mlp, hidden: [8], out: 1, nonsense: 1}\n",
                                           encoding="utf-8")
    assert cli.main(["check", str(record)]) == 1
    assert "nonsense" in capsys.readouterr().out


@pytest.fixture
def two_records(tmp_path):
    first = make_record(tmp_path / "alpha", seed=1, params={"lr": 0.01})
    second = make_record(tmp_path / "beta", seed=2, params={"lr": 0.001}, turns=5)
    return first, second


def test_summarize_prints_the_table_and_the_best_line(two_records, capsys):
    first, second = two_records
    assert cli.main(["summarize", str(first), str(second)]) == 0
    out = capsys.readouterr().out
    assert "alpha" in out and "beta" in out
    assert out.splitlines()[0].startswith("name  ")
    best = [line for line in out.splitlines() if line.startswith("best:")]
    assert len(best) == 1
    found = summary.summarize([str(first), str(second)])
    assert found.best["name"] in best[0] and "val/rmse" in best[0] and "(min)" in best[0]
    assert format(found.best["best"], ".4g") in best[0] and f"at turn {found.best['best_turn']}" in best[0]


def test_summarize_by_mode_and_top(two_records, capsys):
    first, second = two_records
    assert cli.main(["summarize", str(first), str(second), "--by", "val/mse", "--mode", "max", "--top", "1"]) == 0
    out = capsys.readouterr().out
    best = [line for line in out.splitlines() if line.startswith("best:")][0]
    assert "val/mse" in best and "(max)" in best
    expected = summary.summarize([str(first), str(second)], by="val/mse", mode="max").best["name"]
    other = "beta" if expected == "alpha" else "alpha"
    assert expected in out and other not in out


def test_summarize_by_a_series_no_record_logs_says_so_instead_of_naming_one(two_records, capsys):
    first, second = two_records
    assert cli.main(["summarize", str(first), str(second), "--by", "val/nothing"]) == 0
    out = capsys.readouterr().out
    assert "alpha" in out and "beta" in out
    best = [line for line in out.splitlines() if line.startswith("best:")]
    assert best == ["best: none of the records has a value for val/nothing"]


def test_summarize_markdown_prints_a_table_header(two_records, capsys):
    first, second = two_records
    assert cli.main(["summarize", str(first), str(second), "--markdown"]) == 0
    out = capsys.readouterr().out
    header = out.splitlines()[0]
    assert header.startswith("|") and header.count("|") >= 3 and "name" in header
    assert "---" in out.splitlines()[1]
    assert "alpha" in out and "beta" in out
    assert not any(line.startswith("best:") for line in out.splitlines())


def test_summarize_out_writes_csv_and_markdown(two_records, tmp_path, capsys):
    first, second = two_records
    out_dir = tmp_path / "report"
    assert cli.main(["summarize", str(first), str(second), "--out", str(out_dir)]) == 0
    out = capsys.readouterr().out
    csv = out_dir / "summary.csv"
    markdown = out_dir / "summary.md"
    assert csv.exists() and markdown.exists()
    assert str(csv) in out and str(markdown) in out
    table = pandas.read_csv(csv)
    assert set(table["name"]) == {"alpha", "beta"}
    assert "|" in markdown.read_text(encoding="utf-8")


def test_summarize_unknown_path_exits_2(two_records, tmp_path, capsys):
    first, _ = two_records
    assert cli.main(["summarize", str(first), str(tmp_path / "nowhere")]) == 2
    assert capsys.readouterr().err.startswith("error:")


def test_summarize_rejects_a_bad_mode(two_records):
    first, _ = two_records
    with pytest.raises(SystemExit) as raised:
        cli.main(["summarize", str(first), "--mode", "median"])
    assert raised.value.code == 2


def test_summarize_needs_a_record():
    with pytest.raises(SystemExit) as raised:
        cli.main(["summarize"])
    assert raised.value.code == 2


def test_summarize_columns_picks_the_printed_columns(two_records, capsys):
    first, second = two_records
    assert cli.main(["summarize", str(first), str(second)]) == 0
    narrow = capsys.readouterr().out.splitlines()[0].split()
    assert "last_val/rmse" in narrow and "last_train/mse" not in narrow
    assert cli.main(["summarize", str(first), str(second), "--columns", "all"]) == 0
    wide = capsys.readouterr().out.splitlines()[0].split()
    assert "last_train/mse" in wide and "last_lr/main" in wide
    assert cli.main(["summarize", str(first), str(second), "--columns", "name,best"]) == 0
    assert capsys.readouterr().out.splitlines()[0].split() == ["name", "best"]
    assert cli.main(["summarize", str(first), "--markdown", "--columns", "name, status"]) == 0
    assert capsys.readouterr().out.splitlines()[0] == "| name | status |"
    assert cli.main(["summarize", str(first), "--columns", "nothing"]) == 2
    assert "unknown columns: nothing" in capsys.readouterr().err
