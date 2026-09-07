"""Config 12: config 01 as a lower layer with record overridden, then resume from the record."""

from pathlib import Path

import pytest

import kalfa  # noqa: F401
from conftest import ROOT
from kalfa.api import check, resume, run
from kalfa.config import parse_sets
from kalfa.record import read_history, resume_chain

CONFIG = str(ROOT / "configs" / "12_resume.yaml")


def test_layers_and_source_comments(workdir):
    prepared = check([CONFIG])
    assert prepared.problems == []
    text = prepared.surface.layers_text()
    assert "01_mlp_regression.yaml (included by 12_resume.yaml)" in text and "overrides record" in text
    assert "tabular.yaml (included by 01_mlp_regression.yaml)" in text
    assert [path for path, _, _ in prepared.surface.overrides] == [("record",)]


def test_run_then_resume(workdir):
    result = run([CONFIG], parse_sets(params=["epochs=2"]), when="first")
    assert result.record == "runs/housing_resume_first"
    resolved = (Path(result.record) / "resolved.yaml").read_text()
    assert "record: runs/housing_resume_$datetime$  # " in resolved and "12_resume.yaml" in resolved
    assert "01_mlp_regression.yaml" in resolved.split("record:")[1].splitlines()[0]
    assert "epochs: 2  # --set overrides" in resolved
    continued = resume(result.record, parse_sets(["training.epochs=4"]), when="second")
    assert continued.record == "runs/housing_resume_second"
    history = read_history(continued.record)
    assert [line["turn"] for line in history] == [3, 4]
    assert resume_chain(continued.record) == [result.record]
    assert (Path(continued.record) / "checkpoints" / "best.pt").exists()
    with pytest.raises(Exception, match="nothing to resume"):
        resume("runs/nowhere")
