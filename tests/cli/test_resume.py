import json
import re
import shutil

import pytest

from kalfa.cli import main
from kalfa.record import resume_chain
from kalfa.std.common.history import History


@pytest.fixture
def copy(reference, tmp_path):
    target = tmp_path / "ref"
    shutil.copytree(reference.record, target)
    return str(target)


def test_resume_continues_turns_four_and_five_into_a_new_record(copy, tmp_path, capsys):
    target = tmp_path / "resumed"
    assert main(["resume", copy, "--set", "training.epochs=5", "--set", f"record={target}", "--no-progress"]) == 0
    out = capsys.readouterr().out
    assert re.fullmatch(rf"resumed r_[0-9a-f]{{8}}: ok; device cpu; record {re.escape(str(target))}\n", out)
    history = History.read(target)
    assert [line["turn"] for line in history] == [4, 5] and [line["global_step"] for line in history] == [40, 50]
    assert json.loads((target / "resume.json").read_text()) == {"resume_from": copy,
                                                                 "checkpoint": f"{copy}/checkpoints/last.pt"}
    assert resume_chain(target) == [copy]
    assert json.loads((target / "manifest.json").read_text())["kind"] == "run"
    assert json.loads((target / "run.json").read_text())["status"] == "ok"
    assert sorted(path.name for path in (target / "checkpoints").iterdir()) == ["best.pt", "last.pt"]
    assert (target / "final" / "state.pt").exists() and (target / "predictions.parquet").exists()
    assert "training:" in (target / "resolved.yaml").read_text()
    assert "  epochs: 5" in (target / "resolved.yaml").read_text()
    again = tmp_path / "again"
    assert main(["resume", str(target), "--set", "training.epochs=6", "--set", f"record={again}", "--no-progress"]) == 0
    assert [line["turn"] for line in History.read(again)] == [6]
    assert resume_chain(again) == [str(target), copy]


def test_resume_refuses_a_directory_that_is_no_record(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["resume", str(empty)]) == 1
    assert capsys.readouterr().err == f"{empty} has neither checkpoints/last.pt nor final/state.pt; nothing to resume\n"
    assert main(["resume", str(tmp_path / "missing")]) == 1
    assert capsys.readouterr().err == (f"{tmp_path / 'missing'} has neither checkpoints/last.pt nor final/state.pt; "
                                       "nothing to resume\n")


def test_resume_refuses_a_record_directory_that_is_not_empty(copy, capsys):
    assert main(["resume", copy, "--set", f"record={copy}"]) == 1
    assert capsys.readouterr().err == f"record directory {copy} exists and is not empty; change record or remove it\n"
