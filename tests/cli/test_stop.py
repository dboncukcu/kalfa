import shutil
import signal

import pytest

from kalfa.cli import Interrupt, main
from kalfa.record import Record
from kalfa.std.common.log import Monitor


def running(root, name):
    record = Record(root / name)
    record.manifest("run", name=name)
    record.append("history.jsonl", {"turn": 1, "global_step": 1, "val/rmse": 1.0, "rules": []})
    return record


def test_stop_refuses_a_finished_record(reference, tmp_path, capsys):
    copy = tmp_path / "ref"
    shutil.copytree(reference.record, copy)
    assert main(["stop", str(copy)]) == 1
    assert capsys.readouterr().err == f"{copy} has ended already (finished); there is nothing to stop\n"
    assert not (copy / "stop.json").exists()


def test_stop_writes_the_request_into_a_running_record(tmp_path, capsys):
    run = running(tmp_path, "one")
    assert main(["stop", str(run.directory)]) == 0
    assert capsys.readouterr().out == f"stop requested for {run.directory}; the run ends after its current turn\n"
    note = run.stop_note()
    assert note["by"] == "cli" and sorted(note) == ["at", "by"]
    assert run.status() == {"state": "running", "last_seen": run.last_seen(), "stop": note}
    assert main(["stop", str(tmp_path / "nowhere")]) == 1
    assert capsys.readouterr().err == (f"{tmp_path / 'nowhere'} is no record directory; a record holds manifest.json "
                                       "or resolved.yaml\n")
    run.write_json("run.json", {"status": "failed"})
    assert main(["stop", str(run.directory)]) == 1
    assert capsys.readouterr().err == f"{run.directory} has ended already (failed); there is nothing to stop\n"


def test_stop_reaches_the_running_points_of_a_sweep_root(tmp_path, capsys):
    sweep = Record(tmp_path / "grid")
    sweep.manifest("sweep", total=4)
    first = running(sweep.directory, "0000")
    done = running(sweep.directory, "0001")
    done.write_json("run.json", {"status": "ok"})
    pending = Record(sweep.directory / "0002")
    pending.manifest("point", id=2)
    second = running(sweep.directory, "0003")
    assert main(["stop", str(sweep.directory)]) == 0
    assert capsys.readouterr().out == (f"stop requested for {sweep.directory} and 2 running points; the run ends after "
                                       "its current turn\n")
    assert sweep.stop_requested() and first.stop_requested() and second.stop_requested()
    assert not done.stop_requested() and not pending.stop_requested()
    assert sweep.stop_note()["by"] == "cli" and first.stop_note()["by"] == "cli"
    done.path("run.json").unlink()
    assert main(["stop", str(sweep.directory), str(done.directory)]) == 0
    assert capsys.readouterr().out == (
        f"stop requested for {sweep.directory} and 3 running points; the run ends after its current turn\n"
        f"stop requested for {done.directory}; the run ends after its current turn\n")


def test_ctrl_c_asks_for_a_stop_while_training_and_aborts_otherwise(tmp_path, capsys):
    previous = signal.getsignal(signal.SIGINT)
    monitor = Monitor()
    with Interrupt(monitor) as interrupt:
        assert signal.getsignal(signal.SIGINT) == interrupt.handle and not interrupt.asked
        with pytest.raises(KeyboardInterrupt):
            interrupt.handle(signal.SIGINT, None)
        monitor.open(tmp_path)
        monitor.sink({"kind": "started", "path": "training.epochs", "total": 3})
        assert monitor.training
        interrupt.handle(signal.SIGINT, None)
        assert interrupt.asked and Record(tmp_path).stop_note()["by"] == "ctrl-c"
        assert capsys.readouterr().err == ("stop requested: the run ends after this turn with its final state, "
                                           "predictions and plots; ctrl-c again aborts it now\n")
        with pytest.raises(KeyboardInterrupt):
            interrupt.handle(signal.SIGINT, None)
        monitor.sink({"kind": "finished", "path": "training.epochs"})
        assert not monitor.training
    assert signal.getsignal(signal.SIGINT) == previous
    monitor.close()
