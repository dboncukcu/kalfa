from pathlib import Path

import pytest

from kalfa.api import check, resume, run
from kalfa.config import parse_sets
from kalfa.record import read_history, resume_chain

pytestmark = pytest.mark.slow


def test_layers_and_source_comments(dataset):
    dataset("12_resume")
    prepared = check(["config.yaml"])
    assert prepared.problems == []
    text = prepared.surface.layers_text()
    assert "included by" in text and "overrides record" in text
    assert "tabular.yaml (included by" in text
    assert [path for path, _, _ in prepared.surface.overrides] == [("record",)]


def test_run_then_resume(dataset):
    dataset("12_resume")
    result = run(["config.yaml"], parse_sets(params=["epochs=2"]), when="first")
    assert result.record == "runs/housing_resume_first"
    resolved = (Path(result.record) / "resolved.yaml").read_text()
    assert "record: runs/housing_resume_$datetime$  # " in resolved
    assert "epochs: 2  # --set overrides" in resolved
    continued = resume(result.record, parse_sets(["training.epochs=4"]), when="second")
    assert continued.record == "runs/housing_resume_second"
    history = read_history(continued.record)
    assert [line["turn"] for line in history] == [3, 4]
    assert resume_chain(continued.record) == [result.record]
    assert (Path(continued.record) / "checkpoints" / "best.pt").exists()
    with pytest.raises(Exception, match="nothing to resume"):
        resume("runs/nowhere")
