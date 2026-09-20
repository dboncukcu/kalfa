import contextlib
import io
import json
import re
from types import SimpleNamespace

import pandas
import pytest

from data import write_housing
from helpers import config_path, inside
from kalfa.cli import main
from kalfa.record import resume_chain


pytestmark = pytest.mark.slow

TINY = config_path("tiny")
FILES = ["architecture.json", "checkpoints", "contract.yaml", "data.json", "device.json", "events.jsonl", "final",
         "fitted", "flow.yaml", "git.json", "history.jsonl", "host.json", "manifest.json", "plots",
         "predictions.parquet", "resolved.yaml", "run.json", "stderr.txt", "stdout.txt", "steps.jsonl"]


def line_of(path, needle):
    return next(number for number, line in enumerate(path.read_text().splitlines(), 1) if needle in line)


def call(argv):
    with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
        code = main(argv)
    return SimpleNamespace(code=code, out=out.getvalue(), err=err.getvalue())


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    home = tmp_path_factory.mktemp("tiny")
    write_housing(home / "housing.parquet")
    with inside(home):
        result = call(["run", TINY, "--set", "record=runs/tiny", "-p", "seed=5", "--log", "debug", "--progress",
                       "steps", "--log-every", "5"])
    result.home = home
    result.record = home / "runs" / "tiny"
    return result


@pytest.fixture(scope="module")
def resumed(tiny):
    with inside(tiny.home):
        result = call(["resume", "runs/tiny", "--set", "training.epochs=2", "--set", "record=runs/tiny_more", "--log",
                       "--no-progress"])
    result.record = tiny.home / "runs" / "tiny_more"
    return result


@pytest.fixture
def home(tiny, monkeypatch):
    monkeypatch.chdir(tiny.home)
    return tiny.home


def test_run_trains_into_the_record(tiny):
    assert tiny.code == 0
    assert re.fullmatch(r"run r_[0-9a-f]{8}: ok; device cpu; record runs/tiny\n", tiny.out)
    assert sorted(path.name for path in tiny.record.iterdir()) == FILES
    assert json.loads((tiny.record / "run.json").read_text())["status"] == "ok"
    assert [json.loads(line)["turn"] for line in (tiny.record / "history.jsonl").read_text().splitlines()] == [1]
    assert sorted(path.name for path in (tiny.record / "checkpoints").iterdir()) == ["best.pt", "last.pt"]
    assert (tiny.record / "final" / "state.pt").exists()
    assert sorted(path.name for path in (tiny.record / "plots").iterdir()) == ["loss_curve.png"]
    table = pandas.read_parquet(tiny.record / "predictions.parquet")
    assert list(table.columns) == ["row", "price", "raw_y", "pred_y"] and len(table) == 300


def test_run_set_and_param_override_the_config(tiny):
    resolved = (tiny.record / "resolved.yaml").read_text()
    assert re.search(rf"^  seed: 5 +# --set overrides {re.escape(TINY)}:{line_of(tiny.home / TINY, 'params:')}$",
                     resolved, re.M)
    record_line = line_of(tiny.home / TINY, "record:")
    assert re.search(rf"^record: runs/tiny +# --set overrides {re.escape(TINY)}:{record_line}$", resolved, re.M)
    assert "\nseed: 5\n" in resolved
    manifest = json.loads((tiny.record / "manifest.json").read_text())
    assert manifest["kind"] == "run" and manifest["name"] == "tiny" and manifest["turn"] == "epoch"
    assert manifest["params"] == {"epochs": 1, "lr": 0.01, "seed": 5} and manifest["config"] == [TINY]
    assert manifest["prepared"] is None


def test_run_log_debug_narrates_the_nodes_and_the_decisions(tiny):
    assert "  DEBUG  run             seed 5\n" in tiny.err
    assert "  INFO   data.source     reading housing.parquet\n" in tiny.err
    assert "  INFO   data.source     2000 rows, 9 columns (" in tiny.err
    assert "  INFO   data.split      random: train 1400, valid 300, test 300\n" in tiny.err
    assert "  INFO   optimizers      model: adam lr 0.01 over model, loss mse\n" in tiny.err
    assert "  DEBUG  optimizers.build_opt_model  started\n" in tiny.err
    assert "  DEBUG  training.turn   turn 1: 11 steps, model x11\n" in tiny.err
    assert "  INFO   training.ckpt   wrote best.pt, last.pt\n" in tiny.err
    assert "  INFO   training.turn   turn 1  train/mse " in tiny.err
    assert "  INFO   after           predictions.parquet: 300 rows\n" in tiny.err
    assert "  INFO   run             finished in " in tiny.err


def test_run_log_every_prints_a_line_every_n_steps(tiny):
    assert "  INFO   training.step   step 5  loss/model " in tiny.err
    assert "  INFO   training.step   step 10  loss/model " in tiny.err
    assert "  lr/model 0.01" in tiny.err
    assert not re.search(r"training\.step   step [1234679] ", tiny.err)


def test_run_progress_steps_draws_an_inner_bar_over_the_steps(tiny):
    assert "step/s" in tiny.err and "turn/s" in tiny.err and "/11 [" in tiny.err and "1/1 [" in tiny.err


def test_run_refuses_a_record_that_exists_and_is_not_empty(home, capsys):
    assert main(["run", TINY, "--set", "record=runs/tiny"]) == 1
    assert capsys.readouterr().err == "record directory runs/tiny exists and is not empty; change record or remove it\n"


def test_resume_continues_the_record_with_turn_lines_and_no_bar(home, resumed):
    assert resumed.code == 0
    assert re.fullmatch(r"resumed r_[0-9a-f]{8}: ok; device cpu; record runs/tiny_more\n", resumed.out)
    assert "%|" not in resumed.err and "step/s" not in resumed.err
    assert "  INFO   run             resuming runs/tiny from last.pt\n" in resumed.err
    assert "  INFO   training        resuming from runs/tiny/checkpoints/last.pt\n" in resumed.err
    assert "  INFO   training        2 turns, 1 left\n" in resumed.err
    assert "  INFO   training.turn   turn 2  train/mse " in resumed.err
    assert json.loads((resumed.record / "resume.json").read_text()) == {
        "resume_from": "runs/tiny", "checkpoint": "runs/tiny/checkpoints/last.pt"}
    assert [json.loads(line)["turn"] for line in (resumed.record / "history.jsonl").read_text().splitlines()] == [2]
    assert resume_chain(resumed.record) == ["runs/tiny"]


def test_predict_reads_the_record_and_a_new_file(home, capsys):
    assert main(["predict", "runs/tiny"]) == 0
    assert capsys.readouterr().out == "predicted 300 rows with model: runs/tiny/predictions.parquet\n"
    assert main(["predict", "runs/tiny", "--data", "housing.parquet", "--plots"]) == 0
    assert capsys.readouterr().out == ("predicted 2000 rows with model: runs/tiny/predictions_housing.parquet\n"
                                       "plots loss_curve: runs/tiny/plots\n")
    table = pandas.read_parquet(home / "runs" / "tiny" / "predictions_housing.parquet")
    assert list(table.columns) == ["row", "price", "raw_y", "pred_y"] and len(table) == 2000
    assert (home / "runs" / "tiny" / "plots" / "loss_curve_housing.png").exists()


def test_export_writes_the_model_as_pt2(home, capsys):
    assert main(["export", "runs/tiny", "--format", "pt2"]) == 0
    assert capsys.readouterr().out == "exported model as /export/kalfa/pt2: runs/tiny/export/model.pt2\n"
    assert (home / "runs" / "tiny" / "export" / "model.pt2").stat().st_size > 0


def test_plots_redraws_the_loss_curve(home, capsys):
    target = home / "runs" / "tiny" / "plots" / "loss_curve.png"
    target.unlink()
    assert main(["plots", "runs/tiny"]) == 0
    assert capsys.readouterr().out == "plots loss_curve: runs/tiny/plots\n"
    assert target.exists()


def test_collect_tabulates_the_run_and_its_continuation(home, resumed, capsys):
    assert main(["collect", "runs/tiny", "runs/tiny_more", "--out", "summary"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("── RUNS ")
    assert re.search(r"^  dir\s+turns\s+test/mse\s+test/rmse\s+val/mse\s+val/rmse$", out, re.M)
    assert re.search(r"^  runs/tiny\s+1\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+$", out, re.M)
    assert re.search(r"^  runs/tiny_more\s+1\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+$", out, re.M)
    assert out.endswith("\nwrote sweep.csv, sweep.json and sweep.md under summary\n")
    table = json.loads((home / "summary" / "sweep.json").read_text())
    assert table["varying"] == [] and [row["dir"] for row in table["rows"]] == ["runs/tiny", "runs/tiny_more"]
    assert [row["turns"] for row in table["rows"]] == [1, 1]
    assert (home / "summary" / "sweep.md").read_text().startswith(
        "# sweep\n\n| dir | turns | test/mse | test/rmse | val/mse | val/rmse |\n|---|---|---|---|---|---|\n| "
        "runs/tiny | 1 | ")


def test_collect_writes_the_csv_it_announces(home, resumed, capsys):
    assert main(["collect", "runs/tiny", "runs/tiny_more", "--out", "announced"]) == 0
    assert capsys.readouterr().out.endswith("wrote sweep.csv, sweep.json and sweep.md under announced\n")
    assert (home / "announced" / "sweep.csv").exists()


def test_stop_refuses_the_finished_record(home, capsys):
    assert main(["stop", "runs/tiny"]) == 1
    assert capsys.readouterr().err == "runs/tiny has ended already (finished); there is nothing to stop\n"
