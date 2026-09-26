import json
import re
import shutil
from pathlib import Path

import pytest

from data import write_housing
from helpers import config_path
from kalfa.cli import main
from kalfa.std.common.files import read_lines
from kalfa.std.common.history import History


REFERENCE = config_path("reference")
SWEEP = config_path("sweep")


def fail(record):
    summary = json.loads((record / "run.json").read_text())
    (record / "run.json").write_text(json.dumps({**summary, "status": "failed"}))


@pytest.fixture
def failed(reference, tmp_path):
    target = tmp_path / "ref"
    shutil.copytree(reference.record, target)
    fail(target)
    (target / "predictions.parquet").unlink()
    return target


@pytest.fixture
def housing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_housing(tmp_path / "housing.parquet")
    return tmp_path


def attempts_of(record):
    return [(entry["attempt"], entry["stage"], entry["from"], entry["turn"], entry["status"])
            for entry in json.loads((record / "repair.json").read_text())["attempts"]]


def test_repair_runs_the_after_block_of_a_run_that_failed_after_training(failed, capsys):
    assert main(["repair", REFERENCE, "--record", str(failed), "--no-progress"]) == 0
    out = capsys.readouterr().out
    assert re.fullmatch(rf"repaired r_[0-9a-f]{{8}} \(after from turn 3, attempt 1\): ok; device cpu; record "
                        rf"{re.escape(str(failed))}\n", out)
    assert json.loads((failed / "run.json").read_text())["status"] == "ok"
    assert json.loads((failed / "attempts" / "1" / "run.json").read_text())["status"] == "failed"
    assert (failed / "predictions.parquet").exists() and (failed / "final" / "state.pt").exists()
    assert [line["turn"] for line in History.read(failed)] == [1, 2, 3]
    assert attempts_of(failed) == [(1, "after", "final/state.pt", 3, "ok")]
    assert not (failed / "heartbeat").exists() and not (failed / "failure.json").exists()


def test_repair_goes_on_training_from_the_last_checkpoint(failed, capsys):
    shutil.rmtree(failed / "final")
    steps = read_lines(failed / "steps.jsonl")
    History.append(failed, {"turn": 4, "global_step": 40, "rules": []})
    History.append_steps(failed, [{"step": steps[-1]["step"] + 1, "turn": 4}])
    assert main(["repair", REFERENCE, "--record", str(failed), "--no-progress"]) == 0
    assert "(training from turn 3, attempt 1): ok" in capsys.readouterr().out
    assert [line["turn"] for line in History.read(failed)] == [1, 2, 3]
    assert read_lines(failed / "steps.jsonl") == steps
    assert [line["turn"] for line in read_lines(failed / "attempts" / "1" / "history.jsonl")] == [1, 2, 3, 4]
    assert attempts_of(failed) == [(1, "training", "checkpoints/last.pt", 3, "ok")]
    assert json.loads((failed / "run.json").read_text())["status"] == "ok"
    assert (failed / "final" / "state.pt").exists() and (failed / "predictions.parquet").exists()


def test_repair_refuses_what_it_cannot_continue(reference, tmp_path, capsys):
    target = tmp_path / "ref"
    shutil.copytree(reference.record, target)
    assert main(["repair", REFERENCE, "--record", str(target)]) == 1
    assert capsys.readouterr().err == f"{target} is finished, not failed; repair continues failed runs only\n"
    fail(target)
    assert main(["repair", REFERENCE]) == 1
    assert capsys.readouterr().err == ("the config writes a new record directory every run (runs/ref_$datetime$); "
                                       "give the run to repair with --record\n")
    assert main(["repair", str(target)]) == 1
    assert capsys.readouterr().err.startswith("repair takes the config files the run was started with")
    assert main(["repair", REFERENCE, "--record", str(target), "--set", "training.epochs=5"]) == 1
    assert capsys.readouterr().err.startswith(f"{target}: the config differs from the one the run trained with at "
                                              f"training.epochs; repair continues the same run")
    shutil.rmtree(target / "final")
    shutil.rmtree(target / "checkpoints")
    assert main(["repair", REFERENCE, "--record", str(target)]) == 1
    assert capsys.readouterr().err.startswith(f"{target} failed before its first checkpoint, so there is nothing "
                                              f"to continue from")
    assert not (target / "attempts").exists() and not (target / "repair.json").exists()


def test_repair_mends_a_failed_point_and_the_loop_skips_the_rest(housing, capsys):
    assert main(["sweep", SWEEP, "--id", "1", "-p", "epochs=1", "--record", "runs/mend", "--no-progress"]) == 0
    capsys.readouterr()
    point = Path("runs/mend/0001")
    fail(point)
    (point / "sweep.json").unlink()
    assert main(["repair", SWEEP, "--id", "1", "-p", "epochs=1", "--record", "runs/mend", "--no-progress"]) == 0
    out = capsys.readouterr().out
    assert re.fullmatch(r"point 1 repaired \(after, attempt 1\): val/rmse=[\d.]+ at turn 1; record runs/mend/0001\n",
                        out)
    entry = json.loads((point / "sweep.json").read_text())
    assert entry["id"] == 1 and entry["point"] == {"lr": 0.01, "width": 32}
    assert json.loads((point / "run.json").read_text())["status"] == "ok"
    assert main(["repair", SWEEP, "-p", "epochs=1", "--record", "runs/mend", "--no-progress"]) == 0
    out = capsys.readouterr().out
    assert "runs/mend: 0 failed point(s) to repair; skipped 1 finished; 3 not started\n" in out
    assert "0/0 failed points repaired under runs/mend" in out
    with pytest.raises(SystemExit) as failure:
        main(["repair", REFERENCE, "--id", "1"])
    assert failure.value.code == 2
    assert capsys.readouterr().err == "kalfa: error: --id needs a config with a sweep section\n"
