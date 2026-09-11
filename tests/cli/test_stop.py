"""kalfa stop and the ctrl-c handler: a stop request is a file in the record."""

import pytest

from kalfa.cli import Interrupt, main
from kalfa.record import Record
from kalfa.std.common.log import Monitor


def running(root, name):
    record = Record(root / name)
    record.manifest("run", name=name)
    record.append("history.jsonl", {"turn": 1, "global_step": 1, "val/rmse": 1.0, "rules": []})
    return record


def test_stop_writes_the_request_into_running_records_only(tmp_path, capsys):
    run = running(tmp_path, "one")
    assert main(["stop", str(run.directory)]) == 0
    assert "stop requested" in capsys.readouterr().out and run.stop_note()["by"] == "cli"
    run.write_json("run.json", {"status": "ok"})
    assert main(["stop", str(run.directory)]) == 1
    assert "ended already" in capsys.readouterr().err
    assert main(["stop", str(tmp_path / "nowhere")]) == 1
    assert "no record directory" in capsys.readouterr().err
    sweep = Record(tmp_path / "grid")
    sweep.manifest("sweep", total=2)
    point = running(tmp_path / "grid", "0000")
    done = running(tmp_path / "grid", "0001")
    done.write_json("run.json", {"status": "ok"})
    assert main(["stop", str(sweep.directory)]) == 0
    assert "and 1 running point;" in capsys.readouterr().out
    assert sweep.stop_requested() and point.stop_requested() and not done.stop_requested()


def test_ctrl_c_asks_for_a_stop_during_training_and_aborts_otherwise(tmp_path, capsys):
    monitor = Monitor()
    with Interrupt(monitor) as interrupt:
        with pytest.raises(KeyboardInterrupt):
            interrupt.handle(2, None)
        monitor.open(tmp_path)
        monitor.sink({"kind": "started", "path": "training.epochs", "total": 3})
        interrupt.handle(2, None)
        assert Record(tmp_path).stop_note()["by"] == "ctrl-c" and "stop requested" in capsys.readouterr().err
        with pytest.raises(KeyboardInterrupt):
            interrupt.handle(2, None)
    monitor.close()
